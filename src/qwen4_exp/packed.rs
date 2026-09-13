//! Zero-copy views over the packed qwen4-exp layout (`pack.rs`).

use super::{Manifest, Qwen4ExpConfig};
use crate::quant::{GROUP_SIZE, QLinear};
use anyhow::{Context, Result};
use half::bf16;
use memmap2::Mmap;
use std::path::{Path, PathBuf};

pub struct Packed {
    pub cfg: Qwen4ExpConfig,
    pub manifest: Manifest,
    pub dir: PathBuf,
    pub dense: Mmap,
    pub experts: Mmap,
    /// Open handle on experts.bin for advisory reads (prefetch).
    pub experts_file: std::fs::File,
    pub ngram: Mmap,
}

/// One routed expert's three projections, viewed inside its record.
pub struct ExpertRef<'a> {
    pub gate: QLinear<'a>,
    pub up: QLinear<'a>,
    pub down: QLinear<'a>,
}

/// Hash parameters read from a PLE layer's dense tensors.
#[derive(Debug)]
pub(super) struct NgramMetadata {
    pub multipliers: Vec<i64>,
    pub head_offsets: Vec<u64>,
    pub head_sizes: Vec<u64>,
}

fn map(path: &Path) -> Result<Mmap> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;

    // Safety: packed files are read-only while the process runs.
    unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))
}

fn as_u32(bytes: &[u8]) -> &[u32] {
    assert_eq!(
        bytes.as_ptr() as usize % 4,
        0,
        "u32 view must be 4-byte aligned"
    );

    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u32>(), bytes.len() / 4) }
}

fn as_bf16(bytes: &[u8]) -> &[bf16] {
    assert_eq!(
        bytes.as_ptr() as usize % 2,
        0,
        "bf16 view must be 2-byte aligned"
    );

    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<bf16>(), bytes.len() / 2) }
}

impl Packed {
    /// `model_dir` holds config.json and tokenizer.json; the packed files
    /// live in `model_dir/packed`, or directly in a standalone packed directory.
    pub fn open(model_dir: &Path) -> Result<Self> {
        let dir = if model_dir.join("manifest.json").is_file() {
            model_dir.to_owned()
        } else {
            model_dir.join("packed")
        };
        // Imported checkpoints may disable an absent MTP head in their packed
        // config. Older stores can still use metadata beside the source weights.
        let config_dir = if dir.join("config.json").is_file() {
            &dir
        } else {
            model_dir
        };
        let cfg = Qwen4ExpConfig::load(config_dir)?;
        let manifest = Manifest::load(&dir)?;

        anyhow::ensure!(
            manifest.experts.group == GROUP_SIZE,
            "expert group size must be 64"
        );

        let experts_path = dir.join("experts.bin");

        Ok(Packed {
            cfg,
            manifest,
            dense: map(&dir.join("dense.bin"))?,
            experts: map(&experts_path)?,
            experts_file: std::fs::File::open(&experts_path)
                .with_context(|| format!("opening {}", experts_path.display()))?,
            ngram: map(&dir.join("ngram.bin"))?,
            dir,
        })
    }

    pub fn dense_bytes(&self, name: &str) -> Result<&[u8]> {
        let e = self.manifest.dense(name)?;

        Ok(&self.dense[e.offset as usize..(e.offset + e.nbytes) as usize])
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.manifest.dense(name)?.shape)
    }

    pub fn bf16(&self, name: &str) -> Result<&[bf16]> {
        let e = self.manifest.dense(name)?;

        anyhow::ensure!(e.dtype == "BF16", "{name}: expected BF16, got {}", e.dtype);

        Ok(as_bf16(self.dense_bytes(name)?))
    }

    pub fn i64s(&self, name: &str) -> Result<Vec<i64>> {
        let e = self.manifest.dense(name)?;

        anyhow::ensure!(e.dtype == "I64", "{name}: expected I64, got {}", e.dtype);

        Ok(self
            .dense_bytes(name)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| i64::from_le_bytes(c))
            .collect())
    }

    /// Read a PLE embedding's hash parameters from the dense store for either backend.
    pub(super) fn ngram_metadata(&self, prefix: &str) -> Result<NgramMetadata> {
        Ok(NgramMetadata {
            multipliers: self.i64s(&format!("{prefix}.layer_multipliers"))?,
            head_offsets: self
                .i64s(&format!("{prefix}.ngram_heads_offsets"))?
                .into_iter()
                .map(|value| value as u64)
                .collect(),
            head_sizes: self
                .i64s(&format!("{prefix}.ngram_heads_vocab_sizes"))?
                .into_iter()
                .map(|value| value as u64)
                .collect(),
        })
    }

    /// Affine 4-bit group-64 linear layer `{prefix}.{weight,scales,biases}`.
    pub fn qlinear(&self, prefix: &str) -> Result<QLinear<'_>> {
        let w = self.manifest.dense(&format!("{prefix}.weight"))?;

        anyhow::ensure!(w.dtype == "U32", "{prefix}.weight: expected U32");

        let out_dim = w.shape[0];
        let in_dim = w.shape[1] * 8;
        let s = self.manifest.dense(&format!("{prefix}.scales"))?;

        anyhow::ensure!(
            s.shape == vec![out_dim, in_dim / GROUP_SIZE],
            "{prefix}.scales shape {:?} is not group 64",
            s.shape
        );

        Ok(QLinear {
            out_dim,
            in_dim,
            weight: as_u32(self.dense_bytes(&format!("{prefix}.weight"))?),
            scales: self.bf16(&format!("{prefix}.scales"))?,
            biases: self.bf16(&format!("{prefix}.biases"))?,
        })
    }

    /// Record index for a main decoder layer's expert block.
    pub fn expert_layer(&self, layer: usize) -> usize {
        layer
    }

    /// Record index for MTP layer `i`'s expert block.
    pub fn mtp_expert_layer(&self, i: usize) -> Result<usize> {
        let name = format!("mtp.layers.{i}.mlp.switch_mlp");

        self.manifest
            .experts
            .layer_prefixes
            .iter()
            .position(|p| *p == name)
            .with_context(|| format!("no expert records for {name}"))
    }

    pub fn record_offset(&self, record_layer: usize, expert: usize) -> usize {
        let l = &self.manifest.experts;

        ((record_layer * l.experts + expert) as u64 * l.record_stride) as usize
    }

    pub fn expert(&self, record_layer: usize, expert: usize) -> ExpertRef<'_> {
        let l = &self.manifest.experts;
        let base = self.record_offset(record_layer, expert);
        let rec = &self.experts[base..base + l.record_bytes as usize];
        let sl = |off: u64, len: u64| &rec[off as usize..(off + len) as usize];
        let w_up = (l.inter * l.hidden / 2) as u64;
        let s_up = (l.inter * (l.hidden / l.group) * 2) as u64;
        let s_down = (l.hidden * (l.inter / l.group) * 2) as u64;

        ExpertRef {
            gate: QLinear {
                out_dim: l.inter,
                in_dim: l.hidden,
                weight: as_u32(sl(l.gate_w, w_up)),
                scales: as_bf16(sl(l.gate_s, s_up)),
                biases: as_bf16(sl(l.gate_b, s_up)),
            },
            up: QLinear {
                out_dim: l.inter,
                in_dim: l.hidden,
                weight: as_u32(sl(l.up_w, w_up)),
                scales: as_bf16(sl(l.up_s, s_up)),
                biases: as_bf16(sl(l.up_b, s_up)),
            },
            down: QLinear {
                out_dim: l.hidden,
                in_dim: l.inter,
                weight: as_u32(sl(l.down_w, w_up)),
                scales: as_bf16(sl(l.down_s, s_down)),
                biases: as_bf16(sl(l.down_b, s_down)),
            },
        }
    }

    /// Dequantize one hashed n-gram row using the manifest's width and group size.
    pub fn ngram_row(&self, id: u64, dst: &mut [f32]) {
        let n = &self.manifest.ngram;

        debug_assert_eq!(dst.len(), n.dim);

        let base = (id * n.row_bytes) as usize;
        let rec = &self.ngram[base..base + n.row_bytes as usize];
        let wb = n.weight_bytes as usize;
        let sb = n.scale_bytes as usize;
        let scales = &rec[wb..wb + sb];
        let biases = &rec[wb + sb..wb + 2 * sb];
        let g = n.group;

        for (i, out) in dst.iter_mut().enumerate() {
            let word = u32::from_le_bytes(rec[i / 8 * 4..i / 8 * 4 + 4].try_into().unwrap());
            let q = (word >> (4 * (i % 8))) & 0xF;
            let gi = i / g;
            let s =
                bf16::from_bits(u16::from_le_bytes([scales[2 * gi], scales[2 * gi + 1]])).to_f32();
            let b =
                bf16::from_bits(u16::from_le_bytes([biases[2 * gi], biases[2 * gi + 1]])).to_f32();
            *out = s * q as f32 + b;
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/qwen4_exp/packed.rs"]
mod tests;
