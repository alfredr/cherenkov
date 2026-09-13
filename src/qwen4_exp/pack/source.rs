//! Lazy tensor views let both checkpoint formats use the same packed writer.
//! BF16 conversion holds one quantization group at a time, including when a
//! fused expert tensor spans an entire layer. No converted checkpoint is staged.

use super::affine;
use crate::model::{
    CheckpointConventions, Preparation, TensorRole, describe_raw, qwen::tensor_role,
};
use crate::qwen4_exp::Qwen4ExpConfig;
use crate::tensors::{Dtype, ModelWeights, TensorInfo};
use anyhow::{Context, Result, ensure};
use half::bf16;
use memmap2::Advice;
use std::collections::HashMap;
use std::io::Write;
use std::ops::Range;
use std::path::Path;

#[derive(Clone, Copy)]
enum Part {
    Weight,
    Scale,
    Bias,
}

#[derive(Clone, Copy)]
struct Quantized {
    group: usize,
    part: Part,
    /// Contiguous elements selected from each expert in a fused gate/up tensor.
    segment: usize,
    stride: usize,
    start: usize,
}

#[derive(Clone, Copy)]
enum Conversion {
    Copy,
    AddOne,
    Quantize(Quantized),
}

pub(super) struct Tensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub nbytes: usize,
    source: TensorInfo,
    conversion: Conversion,
}

pub(super) struct Source {
    raw: ModelWeights,
    pub tensors: HashMap<String, Tensor>,
    /// Only BF16 import adjusts metadata; prequantized inputs remain unchanged.
    pub config: Option<serde_json::Value>,
}

impl Source {
    pub fn load(dir: &Path) -> Result<Self> {
        let raw = ModelWeights::load_raw(dir)?;
        let config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let description = describe_raw(&raw.checkpoint, &config)?;

        if let Preparation::Unsupported { reasons } = description.preparation() {
            anyhow::bail!(
                "unsupported checkpoint: {}",
                serde_json::to_string(&reasons)?
            );
        }

        let native = matches!(
            description.format.conventions,
            CheckpointConventions::Qwen4ExpHuggingFace
        );
        let mut source = Self {
            raw,
            tensors: HashMap::new(),
            config: None,
        };

        if !native {
            for (name, info) in &source.raw.tensors {
                source.tensors.insert(name.clone(), Tensor::copy(info));
            }

            return Ok(source);
        }

        let cfg = Qwen4ExpConfig::load(dir)?;
        let tensors = source.raw.tensors.clone();

        for (name, info) in tensors {
            source
                .import(&name, &info)
                .with_context(|| format!("importing {name}"))?;
        }

        source.configure(dir, &cfg)?;
        eprintln!("BF16 input: quantizing to affine Q4 while packing");

        Ok(source)
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("tensor {name:?} not found"))
    }

    pub fn estimated_bytes(&self) -> u64 {
        self.tensors.values().map(|t| t.nbytes as u64).sum()
    }

    pub fn prefetch(&self, tensor: &Tensor) {
        // Do not fault in a whole BF16 layer just to convert its first group.
        if self.config.is_some() {
            return;
        }

        let t = &tensor.source;
        let _ = self
            .raw
            .checkpoint
            .objects
            .mapping(crate::model::ObjectId(t.shard))
            .expect("validated source object")
            .advise_range(Advice::WillNeed, t.offset, t.nbytes);
    }

    pub fn bytes(&self, tensor: &Tensor) -> Result<&[u8]> {
        ensure!(
            matches!(tensor.conversion, Conversion::Copy),
            "tensor requires conversion"
        );

        Ok(self.raw.tensor_bytes(&tensor.source))
    }

    pub fn write(&self, tensor: &Tensor, range: Range<usize>, out: &mut impl Write) -> Result<()> {
        ensure!(
            range.start <= range.end && range.end <= tensor.nbytes,
            "tensor write out of bounds"
        );

        let bytes = self.raw.tensor_bytes(&tensor.source);

        match tensor.conversion {
            Conversion::Copy => out.write_all(&bytes[range])?,
            Conversion::AddOne => write_norm(bytes, range, out)?,
            Conversion::Quantize(q) => q.write(bytes, range, out)?,
        }

        Ok(())
    }

    fn insert(&mut self, name: String, tensor: Tensor) -> Result<()> {
        ensure!(
            !self.tensors.contains_key(&name),
            "duplicate normalized tensor {name}"
        );
        self.tensors.insert(name, tensor);

        Ok(())
    }

    fn import(&mut self, name: &str, info: &TensorInfo) -> Result<()> {
        let Some(name) = normalized_name(name) else {
            return Ok(());
        };

        ensure!(
            matches!(info.dtype, Dtype::BF16 | Dtype::I64),
            "native import expects BF16 weights or I64 metadata"
        );
        ensure!(info.shape.iter().all(|&n| n > 0), "empty tensor");

        if let Some(prefix) = name.strip_suffix(".experts.gate_up_proj") {
            return self.fused_experts(prefix, info);
        }

        if let Some(prefix) = name.strip_suffix(".experts.down_proj") {
            ensure!(
                info.shape.len() == 3,
                "expected [experts, hidden, intermediate]"
            );

            return self.quantized(
                &format!("{prefix}.switch_mlp.down_proj"),
                info,
                &info.shape,
                0,
                1,
            );
        }

        if quantized_weight(&name, info) {
            return self.quantized(name.trim_end_matches(".weight"), info, &info.shape, 0, 1);
        }

        let mut tensor = Tensor::copy(info);

        if folded_norm(&name) {
            ensure!(
                info.dtype == Dtype::BF16 && info.shape.len() == 1,
                "expected BF16 norm vector"
            );

            tensor.conversion = Conversion::AddOne;
        }

        self.insert(name, tensor)
    }

    fn fused_experts(&mut self, prefix: &str, info: &TensorInfo) -> Result<()> {
        ensure!(
            info.shape.len() == 3 && info.shape[1].is_multiple_of(2),
            "expected [experts, 2 * intermediate, hidden]"
        );

        let shape = [info.shape[0], info.shape[1] / 2, info.shape[2]];

        for (half, projection) in ["gate_proj", "up_proj"].iter().enumerate() {
            self.quantized(
                &format!("{prefix}.switch_mlp.{projection}"),
                info,
                &shape,
                half,
                2,
            )?;
        }

        Ok(())
    }

    fn quantized(
        &mut self,
        prefix: &str,
        info: &TensorInfo,
        shape: &[usize],
        half: usize,
        halves: usize,
    ) -> Result<()> {
        ensure!(info.dtype == Dtype::BF16, "expected BF16 matrix");

        let width = *shape.last().context("matrix shape missing")?;
        let group = if prefix.contains(".ngram_embedding.") {
            [64, 32, 16, 8]
                .into_iter()
                .find(|g| width.is_multiple_of(*g))
                .context("n-gram width must be divisible by 8")?
        } else {
            ensure!(
                width.is_multiple_of(64),
                "matrix width {width} must be divisible by 64"
            );

            64
        };
        let segment = shape[shape.len() - 2] * width;

        for (suffix, part, dtype, divisor) in [
            ("weight", Part::Weight, Dtype::U32, 8),
            ("scales", Part::Scale, Dtype::BF16, group),
            ("biases", Part::Bias, Dtype::BF16, group),
        ] {
            let mut output_shape = shape.to_vec();
            *output_shape.last_mut().unwrap() /= divisor;
            let nbytes = output_shape.iter().product::<usize>() * dtype.size();
            let tensor = Tensor {
                dtype,
                shape: output_shape,
                nbytes,
                source: info.clone(),
                conversion: Conversion::Quantize(Quantized {
                    group,
                    part,
                    segment,
                    stride: segment * halves,
                    start: segment * half,
                }),
            };

            self.insert(format!("{prefix}.{suffix}"), tensor)?;
        }

        Ok(())
    }

    fn configure(&mut self, dir: &Path, cfg: &Qwen4ExpConfig) -> Result<()> {
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let has_mtp = self.tensors.keys().any(|name| name.starts_with("mtp."));

        if has_mtp {
            ensure!(
                cfg.mtp_num_hidden_layers > 0,
                "MTP weights present but config has no MTP layers"
            );
            self.tensor("mtp.fc_embedding.weight")?;
            self.tensor("mtp.fc_hidden.weight")?;
        } else {
            let text = if config.get("text_config").is_some() {
                &mut config["text_config"]
            } else {
                &mut config
            };
            text["mtp_num_hidden_layers"] = 0.into();

            if let Some(mtp) = text.get_mut("mtp") {
                mtp["num_hidden_layers"] = 0.into();
            }

            eprintln!("no MTP weights found; the packed model will use ordinary decode");
        }

        self.config = Some(config);

        Ok(())
    }
}

impl Tensor {
    fn copy(source: &TensorInfo) -> Self {
        Self {
            dtype: source.dtype,
            shape: source.shape.clone(),
            nbytes: source.nbytes,
            source: source.clone(),
            conversion: Conversion::Copy,
        }
    }
}

impl Quantized {
    fn write(self, source: &[u8], range: Range<usize>, out: &mut impl Write) -> Result<()> {
        let size = match self.part {
            Part::Weight => self.group / 2,
            _ => 2,
        };

        ensure!(
            range.start.is_multiple_of(size) && range.end.is_multiple_of(size),
            "unaligned quantization group"
        );

        for index in range.start / size..range.end / size {
            let element = index * self.group;
            let offset =
                (self.start + element / self.segment * self.stride + element % self.segment) * 2;
            let bytes = source
                .get(offset..offset + self.group * 2)
                .context("fused tensor range out of bounds")?;
            let group = affine::quantize(bytes)?;
            let data = match self.part {
                Part::Weight => &group.weight[..size],
                Part::Scale => &group.scale,
                Part::Bias => &group.bias,
            };

            out.write_all(data)?;
        }

        Ok(())
    }
}

fn normalized_name(name: &str) -> Option<String> {
    if let Some(suffix) = name.strip_prefix("model.language_model.") {
        return Some(format!("language_model.model.{suffix}"));
    }

    if name == "lm_head.weight" {
        return Some("language_model.lm_head.weight".into());
    }

    if name.starts_with("mtp.") {
        return Some(name.into());
    }

    if name.starts_with("model.visual.") || name.starts_with("vision_tower.") {
        return None;
    }

    // HF text-only checkpoints omit the outer multimodal language_model.
    name.strip_prefix("model.")
        .map(|suffix| format!("language_model.model.{suffix}"))
}

fn quantized_weight(name: &str, info: &TensorInfo) -> bool {
    matches!(
        tensor_role(name, info.shape.len()),
        TensorRole::Projection | TensorRole::Embedding | TensorRole::NgramEmbedding
    )
}

fn folded_norm(name: &str) -> bool {
    [
        "hc_norm",
        "q_norm",
        "k_norm",
        "q_layernorm",
        "k_layernorm",
        "norm_key",
        "norm_query",
        "norm_conv",
    ]
    .iter()
    .any(|norm| name.ends_with(&format!(".{norm}.weight")))
}

fn write_norm(bytes: &[u8], range: Range<usize>, out: &mut impl Write) -> Result<()> {
    ensure!(
        range.start.is_multiple_of(2) && range.end.is_multiple_of(2),
        "unaligned BF16 norm"
    );

    for pair in bytes[range].as_chunks::<2>().0 {
        let value = bf16::from_bits(u16::from_le_bytes(*pair)).to_f32();

        ensure!(value.is_finite(), "non-finite norm weight");
        out.write_all(&bf16::from_f32(1.0 + value).to_le_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
#[path = "../../../tests/unit/qwen4_exp/pack/source.rs"]
mod tests;
