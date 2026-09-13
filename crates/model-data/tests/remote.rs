//! Opt-in checks against pinned, complete published tensor headers. These read
//! metadata only: they neither download nor verify weight payloads.
#[path = "support/hub.rs"]
mod hub;

use anyhow::{Context, Result, ensure};
use cherenkov_model_data::{
    ContainerFormat, Dtype, Inventory, ObjectId, TensorEncoding, mlx, read_safetensors,
};
use serde_json::json;
use std::collections::{BTreeMap, HashSet};

fn inspect(repo: &str, revision: &str, architecture: &str, affine: bool) -> Result<()> {
    let hub = hub::Hub::open(repo, revision)?;

    ensure!(
        hub.config["model_type"] == architecture,
        "architecture changed"
    );

    let mut tensors = Vec::new();
    let mut names = HashSet::new();
    let mut shards = Vec::new();

    for id in 0..hub.files.len() {
        let source = hub.shard(id)?;
        let inventory = read_safetensors(&source, ObjectId(id))?;

        ensure!(!inventory.tensors.is_empty(), "empty shard");

        for tensor in &inventory.tensors {
            ensure!(
                names.insert(tensor.name.clone()),
                "duplicate tensor {}",
                tensor.name
            );
            tensor.validate()?;

            if let Some(index) = hub.index["weight_map"].as_object() {
                ensure!(
                    index.get(&tensor.name).and_then(|v| v.as_str())
                        == Some(hub.files[id].as_str()),
                    "shard index mismatch"
                );
            }
        }

        tensors.extend(inventory.tensors);
        shards.push(inventory.metadata);
        eprintln!("{repo}: shard {}/{}", id + 1, hub.files.len());
    }

    if let Some(index) = hub.index["weight_map"].as_object() {
        ensure!(
            names.len() == index.len(),
            "index entries missing from shards"
        );
    }

    let inventory = Inventory {
        format: ContainerFormat::Safetensors,
        metadata: json!({"config": hub.config, "index": hub.index, "shards": shards}),
        tensors,
    };
    let described = if affine {
        mlx::tensors(&inventory, &hub.config)?
    } else {
        inventory.tensors.clone()
    };
    let mut encodings = BTreeMap::new();

    for tensor in &described {
        tensor.validate()?;

        let kind = match &tensor.encoding {
            TensorEncoding::Dense { tensor } => format!("{:?}", tensor.dtype),
            TensorEncoding::Affine {
                bits, group_size, ..
            } => format!("affine_q{bits}_g{group_size}"),
            TensorEncoding::Opaque { name, .. } => name.clone(),
            TensorEncoding::Ggml { .. } => anyhow::bail!("unexpected GGML tensor"),
        };
        *encodings.entry(kind).or_insert(0) += 1;
    }

    if affine {
        ensure!(
            described
                .iter()
                .any(|t| matches!(t.encoding, TensorEncoding::Affine { bits: 4, .. })),
            "no affine Q4 tensors"
        );
    } else {
        ensure!(described.iter().any(|t| matches!(&t.encoding, TensorEncoding::Dense { tensor } if tensor.dtype == Dtype::Bf16)), "no BF16 tensors");
    }

    let roundtrip: Inventory = serde_json::from_value(serde_json::to_value(&inventory)?)?;

    ensure!(
        roundtrip.tensors.len() == inventory.tensors.len(),
        "inventory roundtrip failed"
    );
    eprintln!(
        "{repo}@{revision}: {} stored tensors, {} described tensors; {encodings:?}",
        inventory.tensors.len(),
        described.len()
    );

    Ok(())
}

#[test]
#[ignore = "reads published Hugging Face headers"]
fn qwen4_mlx() -> Result<()> {
    inspect(
        "Sawfwair/Qwen3.8-Flash-Next-MLX-4bit",
        "6cc9bbc0fae9ce26b7670b3ed1e26d557c154506",
        "qwen4_exp",
        true,
    )
    .context("MLX Qwen4-exp")
}

#[test]
#[ignore = "reads published Hugging Face headers"]
fn qwen4_standard() -> Result<()> {
    inspect(
        "Qwen/Qwen3.8-Flash-Next",
        "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        "qwen4_exp",
        false,
    )
    .context("standard Qwen4-exp")
}

#[test]
#[ignore = "reads published Hugging Face headers"]
fn qwen4_small_moe() -> Result<()> {
    inspect(
        "inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B",
        "5cdc1eff790ad299680eda7b97241068224581e4",
        "qwen4_exp",
        false,
    )
    .context("small Qwen4-exp MoE")
}

#[test]
#[ignore = "reads published Hugging Face headers"]
fn qwen27_mlx() -> Result<()> {
    inspect(
        "mlx-community/Qwen3.8-27B-4bit",
        "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
        "qwen3_5",
        true,
    )
    .context("MLX Qwen 27B")
}

#[test]
#[ignore = "reads published Hugging Face headers"]
fn deepseek_flash() -> Result<()> {
    inspect(
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "dba1be0a40aa45a94ad051997016db3960a90277",
        "deepseek_v41",
        false,
    )
    .context("DeepSeek Flash")
}

#[test]
#[ignore = "reads published Hugging Face headers"]
fn deepseek_flash_nvfp4() -> Result<()> {
    inspect(
        "s-zaizen/DeepSeek-V4.1-Flash-NVFP4",
        "179b7cda25486efbaaf8637d696759d9a791d8bd",
        "deepseek_v41",
        false,
    )
    .context("DeepSeek Flash NVFP4")
}
