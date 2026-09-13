use super::qwen::tensor_role as qwen_role;
use super::*;
use crate::qwen4_exp::Manifest;
use anyhow::{Context, Result, ensure};
use cherenkov_model_data as data;
use std::path::Path;

mod packed;

/// A checkpoint description and the immutable objects backing its tensor views.
/// Pass the packed directory explicitly to inspect prepared data.
pub struct Checkpoint {
    /// Architecture, tensor roles, and storage descriptors inferred by the adapter.
    pub description: ModelDescription,
    /// Backing files addressed by the description's object IDs.
    pub objects: MappedObjects,
}

impl Checkpoint {
    /// Inspect a raw checkpoint or a prepared directory containing `manifest.json`.
    /// Backing files must remain unchanged while this checkpoint or its views live.
    pub fn open(path: &Path) -> Result<Self> {
        if path.join("manifest.json").is_file() {
            return packed::open(path);
        }

        let source = data::Checkpoint::open(path)?;
        let description = describe_raw(&source, &source.inventory.metadata["config"])?;

        Ok(Self {
            description,
            objects: source.objects,
        })
    }

    /// Return a retained byte view after checking the span's object ID and bounds.
    /// A view can outlive this checkpoint, but must not outlive its artifact lease.
    pub fn map(&self, span: &DataSpan) -> Result<MappedBytes> {
        self.objects.view(span)
    }
}

pub(crate) fn describe_raw(
    source: &data::Checkpoint,
    config: &serde_json::Value,
) -> Result<ModelDescription> {
    describe_inventory(&source.inventory, &source.objects, config)
}

pub(crate) fn describe_inventory(
    inventory: &data::Inventory,
    source: &dyn ByteSource,
    config: &serde_json::Value,
) -> Result<ModelDescription> {
    let (container, conventions, architecture) = formats(inventory, config);
    let mlx = matches!(
        conventions,
        CheckpointConventions::Qwen4ExpMlx | CheckpointConventions::MlxAffine
    );
    let interpret_mlx = mlx
        || (matches!(architecture, Architecture::Qwen4Exp)
            && matches!(container, ContainerFormat::Safetensors));
    let mut tensors = if interpret_mlx {
        data::mlx::tensors(inventory, config)?
    } else {
        inventory.tensors.clone()
    };

    if matches!(architecture, Architecture::Qwen4Exp)
        && matches!(container, ContainerFormat::Safetensors)
    {
        for tensor in &mut tensors {
            tensor.role = qwen_role(&tensor.name, tensor.shape().map_or(0, <[u64]>::len));
        }
    }

    let ngram = ngram_table(inventory, source, config, &tensors)?;

    Ok(ModelDescription {
        schema_version: 1,
        format: CheckpointFormat {
            container,
            conventions,
        },
        architecture,
        metadata: inventory.metadata.clone(),
        tensors,
        ngram,
    })
}

fn formats(
    inventory: &data::Inventory,
    config: &serde_json::Value,
) -> (ContainerFormat, CheckpointConventions, Architecture) {
    if let data::ContainerFormat::Gguf { version } = inventory.format {
        let name = inventory.metadata["general.architecture"]["value"]
            .as_str()
            .unwrap_or("unknown");
        let architecture = match name {
            "qwen4exp" | "qwen4_exp" => Architecture::Qwen4Exp,
            _ => Architecture::Opaque { name: name.into() },
        };

        return (
            ContainerFormat::Gguf { version },
            CheckpointConventions::Gguf,
            architecture,
        );
    }

    let architecture = architecture(config);
    let native = inventory
        .tensors
        .iter()
        .any(|t| t.name.ends_with(".mlp.experts.gate_up_proj"));
    let mlx = inventory
        .tensors
        .iter()
        .any(|t| t.name.contains(".mlp.switch_mlp."));
    let conventions = match (&architecture, native, mlx) {
        (Architecture::Qwen4Exp, true, _) => CheckpointConventions::Qwen4ExpHuggingFace,
        (Architecture::Qwen4Exp, _, true) => CheckpointConventions::Qwen4ExpMlx,
        _ if declares_mlx(config) => CheckpointConventions::MlxAffine,
        _ => CheckpointConventions::Opaque {
            name: "unrecognized checkpoint conventions".into(),
        },
    };

    (ContainerFormat::Safetensors, conventions, architecture)
}

fn declares_mlx(config: &serde_json::Value) -> bool {
    let Some(quant) = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))
    else {
        return false;
    };

    quant.get("quant_method").is_none()
        && quant["bits"].is_u64()
        && quant["mode"].as_str().unwrap_or("affine") == "affine"
}

fn architecture(config: &serde_json::Value) -> Architecture {
    match config["model_type"].as_str() {
        Some("qwen4_exp" | "qwen4_exp_text") => Architecture::Qwen4Exp,
        name => Architecture::Opaque {
            name: name.unwrap_or("unknown").into(),
        },
    }
}

fn ngram_table(
    inventory: &data::Inventory,
    source: &dyn ByteSource,
    config: &serde_json::Value,
    tensors: &[Tensor],
) -> Result<Option<NgramTable>> {
    let mut entries: Vec<_> = tensors
        .iter()
        .enumerate()
        .filter(|(_, t)| t.role == TensorRole::NgramEmbedding)
        .collect();

    if entries.is_empty() {
        return Ok(None);
    }

    // Numeric shard order matters: lexicographic order places shard_10 before 2.
    entries.sort_by_key(|(_, t)| shard_number(&t.name));

    let name = &entries[0].1.name;
    let prefix = name
        .split(".ngram_embedding.")
        .next()
        .context("n-gram prefix missing")?;
    let read = |suffix| -> Result<Vec<i64>> {
        let entry = inventory
            .tensors
            .iter()
            .find(|t| t.name == format!("{prefix}.{suffix}"))
            .with_context(|| format!("missing n-gram {suffix}"))?;
        let TensorEncoding::Dense { tensor } = &entry.encoding else {
            anyhow::bail!("n-gram metadata must be dense I64");
        };

        ensure!(tensor.dtype == Dtype::I64, "n-gram metadata must be I64");

        ensure!(
            tensor.data.length <= 1024 * 1024,
            "n-gram metadata exceeds 1 MiB"
        );

        let mut bytes = Vec::new();

        source.read(&tensor.data, &mut bytes)?;

        Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&b| i64::from_le_bytes(b))
            .collect())
    };
    let text = config.get("text_config").unwrap_or(config);
    let hashing = NgramHash::Qwen4Exp {
        ngram_size: text["ngram_size"].as_u64().unwrap_or(3),
        heads_per_ngram: text["heads_per_ngram"].as_u64().unwrap_or(8),
        head_offsets: unsigned(read("ngram_heads_offsets")?)?,
        head_vocab_sizes: unsigned(read("ngram_heads_vocab_sizes")?)?,
        layer_multipliers: read("layer_multipliers")?,
    };
    let mut shards = Vec::new();
    let mut first_row = 0_u64;

    for (expected, (id, tensor)) in entries.into_iter().enumerate() {
        ensure!(
            shard_number(&tensor.name) == Some(expected),
            "n-gram shards must be contiguous"
        );

        let rows = *tensor
            .shape()
            .and_then(|s| s.first())
            .context("n-gram rows missing")?;

        shards.push(TableShard {
            first_row,
            rows,
            tensor: TensorId(id),
        });

        first_row = first_row
            .checked_add(rows)
            .context("n-gram row count overflow")?;
    }

    Ok(Some(NgramTable { hashing, shards }))
}

fn unsigned(values: Vec<i64>) -> Result<Vec<u64>> {
    values
        .into_iter()
        .map(|v| u64::try_from(v).context("negative n-gram index metadata"))
        .collect()
}

fn shard_number(name: &str) -> Option<usize> {
    let tail = name.split(".ngram_embedding.").nth(1)?;
    let number = tail
        .strip_prefix("shards.")
        .or_else(|| tail.strip_prefix("shard_"))?;

    number.split('.').next()?.parse().ok()
}
