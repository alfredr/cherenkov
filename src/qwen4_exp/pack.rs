//! Pack qwen4-exp checkpoints into aligned files. MLX Q4 tensors are copied
//! bit for bit; native BF16 tensors are converted as the files are written.
//!
//!   dense.bin    every tensor that is not an expert matrix, an n-gram
//!                shard, or vision (64-byte aligned, name order)
//!   experts.bin  `[layer][expert]` records (see `ExpertLayout`)
//!   ngram.bin    `[row]` records of weight | scales | biases
//!   manifest.json

use super::{DenseEntry, ExpertLayout, Manifest, NgramLayout, PAGE};
use crate::tensors::Dtype;
use source::{Source, Tensor};

mod affine;
mod source;
use crate::units::BYTES_PER_GB;
use anyhow::{Context, Result, ensure};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

const DENSE_ALIGN: u64 = 64;

/// Prepare selected precisions without loading the inference engine. Low-bit
/// stores always derive from the aligned Q4 base, which is retained for reuse.
pub fn prepare(model_dir: &Path, output: Option<&Path>, experts: &[u32]) -> Result<()> {
    ensure!(
        !experts.is_empty() && experts.iter().all(|bits| (2..=4).contains(bits)),
        "select at least one expert precision: 4, 3 or 2"
    );

    let packed_input = model_dir.join("manifest.json").is_file();
    let default_output = if packed_input {
        model_dir.to_owned()
    } else {
        model_dir.join("packed")
    };
    let out = output.unwrap_or(&default_output);

    if !out.join("manifest.json").exists() {
        ensure!(
            !packed_input,
            "a packed input must use its existing store directory"
        );
        pack(model_dir, out)?;
    }

    let manifest = Manifest::load(out)?;
    let source = model_dir.canonicalize()?;

    // An explicit output must not silently select a different checkpoint.
    ensure!(
        out == default_output
            || out.canonicalize()? == source
            || Path::new(&manifest.source_dir).canonicalize().ok() == Some(source),
        "the packed store at {} belongs to a different source model",
        out.display()
    );
    copy_chat_metadata(model_dir, out)?;
    eprintln!("4-bit base ready at {}", out.display());

    let low_bits: Vec<_> = experts.iter().copied().filter(|&bits| bits != 4).collect();

    super::lowbit::ensure_many(out, &manifest.experts, &low_bits)?;
    eprintln!("requested expert stores ready");

    Ok(())
}

/// Auxiliary files copied without replacing metadata already present in a store.
pub(crate) const CHAT_METADATA_FILES: &[&str] = &[
    "chat_template.jinja",
    "tokenizer_config.json",
    "generation_config.json",
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "NOTICE",
    "NOTICE.txt",
    "NOTICE.md",
];

/// Whether an available source has metadata that the prepared store lacks.
pub(crate) fn missing_chat_metadata(source: &Path, target: &Path) -> bool {
    CHAT_METADATA_FILES
        .iter()
        .any(|name| source.join(name).is_file() && !target.join(name).exists())
}

/// Older packed directories can acquire template metadata without repacking weights.
pub(crate) fn copy_chat_metadata(model_dir: &Path, out_dir: &Path) -> Result<()> {
    for name in CHAT_METADATA_FILES {
        let source = model_dir.join(name);
        let target = out_dir.join(name);

        if !source.is_file() || target.exists() {
            continue;
        }

        std::fs::copy(source, target)
            .with_context(|| format!("copying {name} into the packed store"))?;
    }

    Ok(())
}

fn dtype_name(d: Dtype) -> &'static str {
    match d {
        Dtype::U32 => "U32",
        Dtype::F32 => "F32",
        Dtype::F16 => "F16",
        Dtype::BF16 => "BF16",
        Dtype::I64 => "I64",
    }
}

enum Class {
    Dense,
    Expert,
    Ngram,
    Skip,
}

fn classify(name: &str) -> Class {
    if name.starts_with("vision_tower") || name.contains(".visual.") {
        Class::Skip
    } else if name.contains(".mlp.switch_mlp.") {
        Class::Expert
    } else if name.contains(".ngram_embedding.shard") {
        // MLX conversions name these "shards.N" or "shard_N".
        Class::Ngram
    } else {
        Class::Dense
    }
}

struct Progress {
    start: Instant,
    bytes: u64,
    last: Instant,
}

impl Progress {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            bytes: 0,
            last: Instant::now(),
        }
    }

    fn add(&mut self, n: u64, what: &str) {
        self.bytes += n;

        if self.last.elapsed().as_secs_f64() > 10.0 {
            self.last = Instant::now();
            let s = self.start.elapsed().as_secs_f64();

            eprintln!(
                "  {what}: {:.1} GB written, {:.2} GB/s, {:.0} s",
                self.bytes as f64 / BYTES_PER_GB as f64,
                self.bytes as f64 / BYTES_PER_GB as f64 / s,
                s
            );
        }
    }
}

fn pad_to(w: &mut BufWriter<File>, pos: &mut u64, align: u64) -> Result<()> {
    let rem = *pos % align;

    if rem != 0 {
        let pad = (align - rem) as usize;

        w.write_all(&vec![0u8; pad])?;

        *pos += pad as u64;
    }

    Ok(())
}

pub fn pack(model_dir: &Path, out_dir: &Path) -> Result<()> {
    ensure!(
        !out_dir.join("manifest.json").exists(),
        "a packed store already exists at {}; use it for inference",
        out_dir.display()
    );

    let weights = Source::load(model_dir)?;

    crate::storage::create_private_dir(out_dir)?;
    ensure!(
        weights.config.is_none() || model_dir.canonicalize()? != out_dir.canonicalize()?,
        "BF16 import needs a separate output directory to preserve the source configuration"
    );

    // Estimate converted bytes, not BF16 source bytes. The space guard adds
    // 2 GB for alignment and filesystem overhead.
    crate::storage::require_space(out_dir, weights.estimated_bytes())?;

    let mut names: Vec<&String> = weights.tensors.keys().collect();

    names.sort();

    let mut progress = Progress::new();
    let dense = pack_dense(&weights, out_dir, &names, &mut progress)?;
    let layout = pack_experts(&weights, out_dir, &mut progress)?;
    let layout_n = pack_ngram(&weights, out_dir, &names, &mut progress)?;

    // A custom --output directory is itself runnable, without finding its
    // original source directory. Only small runtime metadata is copied.
    if model_dir.canonicalize()? != out_dir.canonicalize()? {
        for name in ["config.json", "tokenizer.json"] {
            std::fs::copy(model_dir.join(name), out_dir.join(name))
                .with_context(|| format!("copying {name} into the packed store"))?;
        }

        copy_chat_metadata(model_dir, out_dir)?;
    }

    if let Some(config) = &weights.config {
        std::fs::write(
            out_dir.join("config.json"),
            serde_json::to_vec_pretty(config)?,
        )?;
    }

    let manifest = Manifest {
        version: 1,
        source_dir: model_dir.display().to_string(),
        dense,
        experts: layout,
        ngram: layout_n,
    };

    std::fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!(
        "packed into {} in {:.0} s ({:.1} GB)",
        out_dir.display(),
        progress.start.elapsed().as_secs_f64(),
        progress.bytes as f64 / BYTES_PER_GB as f64
    );

    Ok(())
}

fn pack_dense(
    weights: &Source,
    out_dir: &Path,
    names: &[&String],
    progress: &mut Progress,
) -> Result<Vec<DenseEntry>> {
    // ---- dense ----
    let mut dense = Vec::new();

    {
        let mut w = BufWriter::with_capacity(8 << 20, File::create(out_dir.join("dense.bin"))?);
        let mut pos = 0u64;

        for name in names {
            if matches!(classify(name), Class::Dense) {
                weights.prefetch(weights.tensor(name)?);
            }
        }

        for name in names {
            if !matches!(classify(name), Class::Dense) {
                continue;
            }

            let t = weights.tensor(name)?;

            pad_to(&mut w, &mut pos, DENSE_ALIGN)?;

            weights.write(t, 0..t.nbytes, &mut w)?;
            dense.push(DenseEntry {
                name: (*name).clone(),
                dtype: dtype_name(t.dtype).into(),
                shape: t.shape.clone(),
                offset: pos,
                nbytes: t.nbytes as u64,
            });

            pos += t.nbytes as u64;

            progress.add(t.nbytes as u64, "dense");
        }

        w.flush()?;
        eprintln!(
            "dense.bin: {} tensors, {:.2} GB",
            dense.len(),
            pos as f64 / BYTES_PER_GB as f64
        );
    }

    Ok(dense)
}

fn pack_experts(weights: &Source, out_dir: &Path, progress: &mut Progress) -> Result<ExpertLayout> {
    // ---- experts ----
    let mut prefixes: Vec<String> = Vec::new();
    let mut layer = 0;

    while weights.tensors.contains_key(&format!(
        "language_model.model.layers.{layer}.mlp.switch_mlp.gate_proj.weight"
    )) {
        prefixes.push(format!(
            "language_model.model.layers.{layer}.mlp.switch_mlp"
        ));

        layer += 1;
    }

    let mut mtp = 0;

    while weights
        .tensors
        .contains_key(&format!("mtp.layers.{mtp}.mlp.switch_mlp.gate_proj.weight"))
    {
        prefixes.push(format!("mtp.layers.{mtp}.mlp.switch_mlp"));

        mtp += 1;
    }

    anyhow::ensure!(!prefixes.is_empty(), "no expert tensors found");

    let gw = weights.tensor(&format!("{}.gate_proj.weight", prefixes[0]))?;
    let gs = weights.tensor(&format!("{}.gate_proj.scales", prefixes[0]))?;
    let dw = weights.tensor(&format!("{}.down_proj.weight", prefixes[0]))?;
    let ds = weights.tensor(&format!("{}.down_proj.scales", prefixes[0]))?;
    let experts = gw.shape[0];
    let inter = gw.shape[1];
    let hidden = gw.shape[2] * 8;

    anyhow::ensure!(
        dw.shape == vec![experts, hidden, inter / 8],
        "down_proj shape {:?}",
        dw.shape
    );

    let group = hidden / gs.shape[2];

    anyhow::ensure!(group == inter / ds.shape[2], "group size mismatch");

    let w_up = (inter * hidden / 2) as u64; // packed 4-bit bytes for gate/up
    let w_down = (hidden * inter / 2) as u64;
    let s_up = (inter * (hidden / group) * 2) as u64;
    let s_down = (hidden * (inter / group) * 2) as u64;
    let layout = {
        let mut off = 0u64;
        let mut take = |n: u64| {
            let o = off;
            off += n;

            o
        };
        let gate_w = take(w_up);
        let up_w = take(w_up);
        let down_w = take(w_down);
        let gate_s = take(s_up);
        let gate_b = take(s_up);
        let up_s = take(s_up);
        let up_b = take(s_up);
        let down_s = take(s_down);
        let down_b = take(s_down);
        let record_bytes = off;

        ExpertLayout {
            layers: prefixes.len(),
            experts,
            inter,
            hidden,
            group,
            record_bytes,
            record_stride: record_bytes.div_ceil(PAGE) * PAGE,
            layer_prefixes: prefixes.clone(),
            gate_w,
            up_w,
            down_w,
            gate_s,
            gate_b,
            up_s,
            up_b,
            down_s,
            down_b,
        }
    };

    eprintln!(
        "experts: {} layers x {} experts, record {} bytes (stride {}), {:.2} GB",
        layout.layers,
        experts,
        layout.record_bytes,
        layout.record_stride,
        (layout.layers * experts) as f64 * layout.record_stride as f64 / BYTES_PER_GB as f64
    );

    {
        let mut w = BufWriter::with_capacity(8 << 20, File::create(out_dir.join("experts.bin"))?);
        let pad = vec![0u8; (layout.record_stride - layout.record_bytes) as usize];
        let mut pos = 0u64;

        const SUFFIXES: [&str; 9] = [
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
            "gate_proj.scales",
            "gate_proj.biases",
            "up_proj.scales",
            "up_proj.biases",
            "down_proj.scales",
            "down_proj.biases",
        ];

        for s in SUFFIXES {
            weights.prefetch(weights.tensor(&format!("{}.{s}", prefixes[0]))?);
        }

        for (pi, prefix) in prefixes.iter().enumerate() {
            if let Some(next) = prefixes.get(pi + 1) {
                for s in SUFFIXES {
                    weights.prefetch(weights.tensor(&format!("{next}.{s}"))?);
                }
            }

            let get = |suffix: &str| weights.tensor(&format!("{prefix}.{suffix}"));
            let parts: [(&Tensor, u64); 9] = [
                (get("gate_proj.weight")?, w_up),
                (get("up_proj.weight")?, w_up),
                (get("down_proj.weight")?, w_down),
                (get("gate_proj.scales")?, s_up),
                (get("gate_proj.biases")?, s_up),
                (get("up_proj.scales")?, s_up),
                (get("up_proj.biases")?, s_up),
                (get("down_proj.scales")?, s_down),
                (get("down_proj.biases")?, s_down),
            ];

            for (buf, per) in &parts {
                anyhow::ensure!(
                    buf.nbytes as u64 == *per * experts as u64,
                    "{prefix}: tensor size {} != {} x {experts}",
                    buf.nbytes,
                    per
                );
            }

            for e in 0..experts {
                for (buf, per) in &parts {
                    let start = (e as u64 * per) as usize;

                    weights.write(buf, start..start + *per as usize, &mut w)?;
                }

                w.write_all(&pad)?;

                pos += layout.record_stride;
            }

            progress.add(experts as u64 * layout.record_stride, prefix);
        }

        w.flush()?;
        eprintln!("experts.bin: {:.2} GB", pos as f64 / BYTES_PER_GB as f64);
    }

    Ok(layout)
}

fn pack_ngram(
    weights: &Source,
    out_dir: &Path,
    names: &[&String],
    progress: &mut Progress,
) -> Result<NgramLayout> {
    // ---- n-gram ----
    let ngram_prefix = names
        .iter()
        .find_map(|n| {
            n.find(".ngram_embedding.")
                .map(|i| n[..i + ".ngram_embedding".len()].to_string())
        })
        .context("n-gram shards not found")?;
    let ple_prefix = ngram_prefix
        .trim_end_matches(".ngram_embedding")
        .to_string();
    // Shard naming differs between converters: "shards.N" or "shard_N".
    let shard = |s: usize, suffix: &str| -> String {
        let dotted = format!("{ngram_prefix}.shards.{s}.{suffix}");

        if weights.tensors.contains_key(&dotted) {
            dotted
        } else {
            format!("{ngram_prefix}.shard_{s}.{suffix}")
        }
    };
    let read_i64 = |name: &str| -> Result<Vec<i64>> {
        let t = weights.tensor(name)?;

        anyhow::ensure!(t.dtype == Dtype::I64, "{name}: expected I64");

        Ok(weights
            .bytes(t)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| i64::from_le_bytes(c))
            .collect())
    };
    let head_offsets = read_i64(&format!("{ple_prefix}.ngram_heads_offsets"))?;
    let head_sizes = read_i64(&format!("{ple_prefix}.ngram_heads_vocab_sizes"))?;
    let multipliers = read_i64(&format!("{ple_prefix}.layer_multipliers"))?;
    let mut shards = 0;

    while weights.tensors.contains_key(&shard(shards, "weight")) {
        shards += 1;
    }

    anyhow::ensure!(shards > 0, "no n-gram shard tensors under {ngram_prefix}");

    let s0w = weights.tensor(&shard(0, "weight"))?;
    let s0s = weights.tensor(&shard(0, "scales"))?;
    let rows_per_shard = s0w.shape[0];
    let dim = s0w.shape[1] * 8;
    let ngroup = dim / s0s.shape[1];
    let weight_bytes = (dim / 2) as u64;
    let scale_bytes = (dim / ngroup * 2) as u64;
    let row_bytes = weight_bytes + 2 * scale_bytes;
    let layout_n = NgramLayout {
        rows: (shards * rows_per_shard) as u64,
        row_bytes,
        dim,
        group: ngroup,
        weight_bytes,
        scale_bytes,
        head_offsets: head_offsets.iter().map(|&v| v as u64).collect(),
        head_vocab_sizes: head_sizes.iter().map(|&v| v as u64).collect(),
        layer_multipliers: multipliers,
    };

    eprintln!(
        "ngram: {shards} shards x {rows_per_shard} rows, dim {dim}, group {ngroup}, row {row_bytes} bytes, {:.2} GB",
        layout_n.rows as f64 * row_bytes as f64 / BYTES_PER_GB as f64
    );

    {
        let mut w = BufWriter::with_capacity(8 << 20, File::create(out_dir.join("ngram.bin"))?);

        for part in ["weight", "scales", "biases"] {
            weights.prefetch(weights.tensor(&shard(0, part))?);
        }

        for s in 0..shards {
            if s + 1 < shards {
                for part in ["weight", "scales", "biases"] {
                    weights.prefetch(weights.tensor(&shard(s + 1, part))?);
                }
            }

            let wt = weights.tensor(&shard(s, "weight"))?;
            let st = weights.tensor(&shard(s, "scales"))?;
            let bt = weights.tensor(&shard(s, "biases"))?;

            anyhow::ensure!(wt.nbytes as u64 == rows_per_shard as u64 * weight_bytes);
            anyhow::ensure!(st.nbytes as u64 == rows_per_shard as u64 * scale_bytes);
            anyhow::ensure!(bt.nbytes == st.nbytes);

            let (wb, sb) = (weight_bytes as usize, scale_bytes as usize);

            for r in 0..rows_per_shard {
                weights.write(wt, r * wb..(r + 1) * wb, &mut w)?;
                weights.write(st, r * sb..(r + 1) * sb, &mut w)?;
                weights.write(bt, r * sb..(r + 1) * sb, &mut w)?;
            }

            progress.add(
                rows_per_shard as u64 * row_bytes,
                &format!("ngram shard {s}"),
            );
        }

        w.flush()?;
    }

    Ok(layout_n)
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4_exp/pack.rs"]
mod tests;
