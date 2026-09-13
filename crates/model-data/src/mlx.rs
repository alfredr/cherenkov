//! MLX affine conventions over a safetensors inventory. This adapter does not
//! interpret architecture-specific names or choose execution precision.
use crate::*;
use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value};
use std::collections::HashMap;

/// Combine MLX weight, scale, and bias entries into logical affine tensors.
/// Per-layer quantization settings override global settings. Unknown encodings
/// retain their metadata and data spans as opaque tensors; no weights are decoded.
pub fn tensors(inventory: &Inventory, config: &Value) -> Result<Vec<Tensor>> {
    let entries: HashMap<_, _> = inventory
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    let mut tensors = Vec::new();

    for tensor in &inventory.tensors {
        if is_component(&entries, &tensor.name) {
            continue;
        }

        let Some(prefix) = tensor.name.strip_suffix(".weight") else {
            tensors.push(tensor.clone());

            continue;
        };
        let TensorEncoding::Dense { tensor: codes } = &tensor.encoding else {
            tensors.push(tensor.clone());

            continue;
        };

        if codes.dtype != Dtype::U32 {
            tensors.push(tensor.clone());

            continue;
        }

        let get = |suffix| -> Result<StoredTensor> {
            let entry = entries
                .get(format!("{prefix}.{suffix}").as_str())
                .with_context(|| format!("{prefix}: missing {suffix}"))?;
            let TensorEncoding::Dense { tensor } = &entry.encoding else {
                anyhow::bail!("{prefix}: unsupported {suffix} storage");
            };

            Ok(tensor.clone())
        };
        let quantization = quantization(config, prefix);
        let bits = quantization["bits"].as_u64();
        let mode = quantization["mode"].as_str().unwrap_or("affine");

        if !matches!(bits, Some(2 | 4 | 8)) || mode != "affine" {
            let mut data = vec![codes.data.clone()];

            for suffix in ["scales", "biases"] {
                if let Some(entry) = entries.get(format!("{prefix}.{suffix}").as_str()) {
                    data.extend(entry.encoding.data().into_iter().cloned());
                }
            }

            tensors.push(Tensor::new(
                tensor.name.clone(),
                tensor.role,
                None,
                TensorEncoding::Opaque {
                    name: "unrecognized MLX weight encoding".into(),
                    metadata: quantization,
                    data,
                },
            ));

            continue;
        }

        let scales = get("scales")?;
        let biases = get("biases")?;
        let bits = bits.context("missing quantization bits")? as u8;

        tensors.push(affine(tensor, codes.clone(), scales, biases, bits)?);
    }

    Ok(tensors)
}

fn affine(
    source: &Tensor,
    codes: StoredTensor,
    scales: StoredTensor,
    biases: StoredTensor,
    bits: u8,
) -> Result<Tensor> {
    let mut shape = codes.shape.clone();
    let axis = shape
        .len()
        .checked_sub(1)
        .context("scalar quantized weight")?;
    shape[axis] = shape[axis]
        .checked_mul(32 / u64::from(bits))
        .context("quantized shape overflow")?;

    ensure!(
        scales.shape.len() == shape.len(),
        "affine scale rank mismatch"
    );

    let groups = scales.shape[axis];

    ensure!(
        groups > 0 && shape[axis].is_multiple_of(groups),
        "invalid affine scale count"
    );

    let encoding = TensorEncoding::Affine {
        bits,
        group_size: shape[axis] / groups,
        group_axis: axis,
        packing: BitPacking::LowFirstU32,
        codes,
        scales,
        offset: AffineOffset::Bias { tensor: biases },
    };
    let tensor = Tensor::new(source.name.clone(), source.role, Some(shape), encoding);

    tensor.validate()?;

    Ok(tensor)
}

fn is_component(entries: &HashMap<&str, &Tensor>, name: &str) -> bool {
    let Some(prefix) = name
        .strip_suffix(".scales")
        .or_else(|| name.strip_suffix(".biases"))
    else {
        return false;
    };

    entries
        .get(format!("{prefix}.weight").as_str())
        .is_some_and(|entry| {
            matches!(
                &entry.encoding, TensorEncoding::Dense { tensor } if tensor.dtype == Dtype::U32
            )
        })
}

fn quantization(config: &Value, prefix: &str) -> Value {
    let Some(root) = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))
    else {
        return Value::Null;
    };
    let mut fields = Map::new();

    for name in ["bits", "group_size", "mode"] {
        if let Some(value) = root.get(name) {
            fields.insert(name.into(), value.clone());
        }
    }

    if let Some(local) = root.get(prefix).and_then(Value::as_object) {
        fields.extend(local.clone());
    }

    Value::Object(fields)
}
