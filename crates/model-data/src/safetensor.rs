use crate::*;
use anyhow::{Context, Result, ensure};
use safetensors::{Dtype as SafeDtype, tensor::Metadata};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Component, Path},
};

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

pub(crate) fn open(dir: &Path) -> Result<Checkpoint> {
    let index = optional_json(&dir.join("model.safetensors.index.json"))?;
    let config = optional_json(&dir.join("config.json"))?;
    let ShardIndex { names, weight_map } = shard_names(&index)?;
    let paths: Vec<_> = names.iter().map(|name| dir.join(name)).collect();
    let objects = MappedObjects::open(paths.iter().map(|p| p.as_path()))?;
    let mut tensors = Vec::new();
    let mut metadata = Vec::new();

    for (id, name) in names.iter().enumerate() {
        let (shard, extra) = inspect(&objects, ObjectId(id))?;

        validate_index(&weight_map, &shard, name)?;
        tensors.extend(shard);
        metadata.push(extra);
    }

    if let Some(index) = &weight_map {
        ensure!(
            tensors.len() == index.len(),
            "safetensors index names missing from shards"
        );
    }

    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    let checkpoint = Checkpoint {
        inventory: Inventory {
            format: ContainerFormat::Safetensors,
            metadata: json!({"config": config, "index": index, "shards": metadata}),
            tensors,
        },
        objects,
    };

    checkpoint.validate()?;

    Ok(checkpoint)
}

/// Read one safetensors header through a byte source. Tensor payloads remain in
/// that source; successful inspection does not claim their bytes were verified.
pub fn read(source: &dyn ByteSource, object: ObjectId) -> Result<Inventory> {
    let (tensors, metadata) = inspect(source, object)?;

    Ok(Inventory {
        format: ContainerFormat::Safetensors,
        metadata: json!({"shards": [metadata]}),
        tensors,
    })
}

fn inspect(source: &dyn ByteSource, object: ObjectId) -> Result<(Vec<Tensor>, Value)> {
    let size = source
        .objects()
        .into_iter()
        .find(|info| info.id == object)
        .context("unknown safetensors object")?
        .bytes;
    let prefix = read_span(source, object, 0, 8)?;
    let header_len = u64::from_le_bytes(
        prefix
            .try_into()
            .map_err(|_| anyhow::anyhow!("truncated header length"))?,
    );

    ensure!(
        header_len <= 64 * 1024 * 1024,
        "safetensors header exceeds 64 MiB"
    );

    let base = 8 + header_len;

    ensure!(base <= size, "safetensors header exceeds object size");

    let header = read_span(source, object, 8, header_len)?;
    // Upstream Metadata validates contiguous offsets, dtype sizes, and shape
    // overflow without requiring a local mapping of the entire weight file.
    let metadata: Metadata = serde_json::from_slice(&header).context("safetensors header")?;
    let end = metadata
        .tensors()
        .values()
        .map(|t| t.data_offsets.1 as u64)
        .max()
        .unwrap_or(0);

    ensure!(
        end.checked_add(base) == Some(size),
        "safetensors payload size mismatch"
    );

    let mut tensors = Vec::new();

    for (name, info) in metadata.tensors() {
        let (start, end) = info.data_offsets;
        let data = DataSpan {
            object,
            offset: base + start as u64,
            length: (end - start) as u64,
        };
        let shape: Vec<_> = info.shape.iter().map(|&n| n as u64).collect();
        let encoding = match dtype(info.dtype) {
            Some(dtype) => TensorEncoding::Dense {
                tensor: StoredTensor::contiguous(dtype, shape.clone(), data)?,
            },
            None => TensorEncoding::Opaque {
                name: info.dtype.to_string(),
                metadata: json!({"dtype": info.dtype.to_string(), "shape": shape}),
                data: vec![data],
            },
        };

        tensors.push(Tensor::new(
            name.clone(),
            TensorRole::Opaque,
            Some(shape),
            encoding,
        ));
    }

    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    Ok((tensors, serde_json::to_value(metadata.metadata())?))
}

fn read_span(
    source: &dyn ByteSource,
    object: ObjectId,
    offset: u64,
    length: u64,
) -> Result<Vec<u8>> {
    let span = DataSpan {
        object,
        offset,
        length,
    };
    let mut bytes = Vec::new();

    source.read(&span, &mut bytes)?;
    ensure!(
        bytes.len() as u64 == length,
        "byte source returned an incomplete span"
    );

    Ok(bytes)
}

fn dtype(dtype: SafeDtype) -> Option<Dtype> {
    Some(match dtype {
        SafeDtype::U32 => Dtype::U32,
        SafeDtype::I64 => Dtype::I64,
        SafeDtype::F32 => Dtype::F32,
        SafeDtype::F16 => Dtype::F16,
        SafeDtype::BF16 => Dtype::Bf16,
        SafeDtype::BOOL => Dtype::Bool,
        SafeDtype::U8 => Dtype::U8,
        SafeDtype::I8 => Dtype::I8,
        SafeDtype::U16 => Dtype::U16,
        SafeDtype::I16 => Dtype::I16,
        SafeDtype::I32 => Dtype::I32,
        SafeDtype::U64 => Dtype::U64,
        SafeDtype::F64 => Dtype::F64,
        _ => return None,
    })
}

struct ShardIndex {
    names: Vec<String>,
    weight_map: Option<BTreeMap<String, String>>,
}

fn shard_names(value: &Value) -> Result<ShardIndex> {
    if value.is_null() {
        return Ok(ShardIndex {
            names: vec!["model.safetensors".into()],
            weight_map: None,
        });
    }

    let index: Index = serde_json::from_value(value.clone()).context("safetensors shard index")?;

    ensure!(!index.weight_map.is_empty(), "empty safetensors index");

    let mut names: Vec<_> = index.weight_map.values().cloned().collect();

    names.sort();
    names.dedup();

    for name in &names {
        ensure!(
            !name.is_empty()
                && Path::new(name)
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
            "shard path must stay inside checkpoint: {name}"
        );
    }

    Ok(ShardIndex {
        names,
        weight_map: Some(index.weight_map),
    })
}

fn validate_index(
    index: &Option<BTreeMap<String, String>>,
    tensors: &[Tensor],
    shard: &str,
) -> Result<()> {
    let Some(index) = index else {
        return Ok(());
    };

    for tensor in tensors {
        ensure!(
            index.get(&tensor.name).is_some_and(|file| file == shard),
            "tensor {} disagrees with shard index",
            tensor.name
        );
    }

    Ok(())
}

fn optional_json(path: &Path) -> Result<Value> {
    match std::fs::read(path) {
        Ok(bytes) => {
            serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}
