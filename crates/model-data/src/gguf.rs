//! Header-only GGUF inspection. Tensor codecs are identified, never treated as
//! interchangeable with MLX affine Q4. Execution/conversion is not implemented.

use super::*;
const BYTES_PER_MIB: usize = 1024 * 1024;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};

pub(crate) fn inspect(map: &[u8]) -> Result<Inventory> {
    let mut reader = Header { bytes: map, pos: 0 };

    ensure!(reader.take(4)? == b"GGUF", "expected a GGUF file");

    let version = reader.u32()?;

    ensure!(
        matches!(version, 2 | 3),
        "unsupported GGUF version {version}"
    );

    let tensor_count = reader.count()?;
    let metadata_count = reader.count()?;
    let mut metadata = serde_json::Map::new();
    let mut alignment = 32_u64;

    for _ in 0..metadata_count {
        let key = reader.string()?.to_owned();
        let kind = reader.u32()?;
        let value = reader.value(kind, 0)?;

        if key == "general.alignment" {
            ensure!(kind == 4, "GGUF alignment must be uint32");

            alignment = value.as_u64().context("invalid GGUF alignment")?;
        }

        ensure!(
            !metadata.contains_key(&key),
            "duplicate GGUF metadata {key}"
        );
        metadata.insert(key, json!({"type": kind, "value": value}));
    }

    let mut tensors = Vec::new();

    for _ in 0..tensor_count {
        let name = reader.string()?.to_owned();
        let rank = reader.u32()?;

        ensure!((1..=4).contains(&rank), "invalid GGUF tensor rank");

        let mut shape = (0..rank)
            .map(|_| reader.u64())
            .collect::<Result<Vec<_>>>()?;

        shape.reverse(); // GGUF lists the fastest-varying dimension first.

        let encoding = codec(reader.u32()?);
        let offset = reader.u64()?;

        tensors.push((name, shape, encoding, offset));
    }

    ensure!(
        alignment > 0 && alignment.is_power_of_two(),
        "invalid GGUF alignment"
    );

    let data_start = (reader.pos as u64)
        .checked_add(alignment - 1)
        .context("GGUF alignment overflow")?
        / alignment
        * alignment;

    ensure!(data_start <= map.len() as u64, "truncated GGUF padding");
    tensors.sort_by_key(|t| t.3);

    let mut described = Vec::new();

    for (i, (name, shape, encoding, relative)) in tensors.iter().enumerate() {
        let offset = data_start
            .checked_add(*relative)
            .context("GGUF tensor offset overflow")?;
        let end = match tensors.get(i + 1) {
            Some(next) => data_start
                .checked_add(next.3)
                .context("GGUF tensor offset overflow")?,
            None => map.len() as u64,
        };

        ensure!(
            relative.is_multiple_of(alignment) && offset <= end && end <= map.len() as u64,
            "GGUF tensor range out of bounds"
        );

        let length = encoded_len(*encoding, shape)?.unwrap_or(end - offset);

        ensure!(length <= end - offset, "truncated GGUF tensor {name}");
        described.push(Tensor::new(
            name.clone(),
            TensorRole::Opaque,
            Some(shape.clone()),
            TensorEncoding::Ggml {
                encoding: *encoding,
                data: DataSpan {
                    object: ObjectId(0),
                    offset,
                    length,
                },
            },
        ));
    }

    Ok(Inventory {
        format: ContainerFormat::Gguf { version },
        metadata: Value::Object(metadata),
        tensors: described,
    })
}

fn encoded_len(encoding: GgmlEncoding, shape: &[u64]) -> Result<Option<u64>> {
    let (block, bytes) = match encoding {
        GgmlEncoding::F32 => (1, 4),
        GgmlEncoding::F16 | GgmlEncoding::Bf16 => (1, 2),
        GgmlEncoding::Q4_0 => (32, 18),
        GgmlEncoding::Q4_1 => (32, 20),
        GgmlEncoding::Q8_0 => (32, 34),
        GgmlEncoding::Q4K => (256, 144),
        GgmlEncoding::Q5K => (256, 176),
        GgmlEncoding::Q6K => (256, 210),
        GgmlEncoding::Opaque { .. } => return Ok(None),
    };
    let width = *shape.last().context("GGUF tensor has no dimensions")?;

    ensure!(
        width > 0 && width.is_multiple_of(block),
        "GGUF row does not fit its quantization block"
    );

    let elements = shape.iter().try_fold(1_u64, |size, &dim| {
        ensure!(dim > 0, "empty GGUF tensor dimension");

        size.checked_mul(dim).context("GGUF tensor size overflow")
    })?;

    Ok(Some(
        (elements / block)
            .checked_mul(bytes)
            .context("GGUF tensor size overflow")?,
    ))
}

fn codec(code: u32) -> GgmlEncoding {
    match code {
        0 => GgmlEncoding::F32,
        1 => GgmlEncoding::F16,
        2 => GgmlEncoding::Q4_0,
        3 => GgmlEncoding::Q4_1,
        8 => GgmlEncoding::Q8_0,
        12 => GgmlEncoding::Q4K,
        13 => GgmlEncoding::Q5K,
        14 => GgmlEncoding::Q6K,
        30 => GgmlEncoding::Bf16,
        code => GgmlEncoding::Opaque { code },
    }
}

struct Header<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Header<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(count)
            .context("GGUF header overflow")?;

        ensure!(end <= 64 * BYTES_PER_MIB, "GGUF header exceeds 64 MiB");

        let bytes = self
            .bytes
            .get(self.pos..end)
            .context("truncated GGUF header")?;
        self.pos = end;

        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn count(&mut self) -> Result<u64> {
        let count = self.u64()?;

        ensure!(
            count <= 1_000_000,
            "GGUF entry count exceeds inspection limit"
        );

        Ok(count)
    }

    fn string(&mut self) -> Result<&'a str> {
        let len = usize::try_from(self.u64()?).context("GGUF string size overflow")?;

        std::str::from_utf8(self.take(len)?).context("invalid UTF-8 in GGUF header")
    }

    fn value(&mut self, kind: u32, depth: usize) -> Result<Value> {
        ensure!(depth <= 8, "GGUF metadata nesting limit exceeded");

        Ok(match kind {
            0 => json!(self.take(1)?[0]),
            1 => json!(self.take(1)?[0] as i8),
            2 => json!(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => json!(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => json!(self.u32()?),
            5 => json!(self.u32()? as i32),
            6 => {
                let bits = self.u32()?;

                float_value(f64::from(f32::from_bits(bits)), u64::from(bits))
            }
            7 => {
                let value = self.take(1)?[0];

                ensure!(value <= 1, "invalid GGUF boolean");

                json!(value == 1)
            }
            8 => json!(self.string()?),
            9 => {
                let element = self.u32()?;
                let count = self.count()?;
                let values = (0..count)
                    .map(|_| self.value(element, depth + 1))
                    .collect::<Result<Vec<_>>>()?;

                json!({"element_type": element, "values": values})
            }
            10 => json!(self.u64()?),
            11 => json!(self.u64()? as i64),
            12 => {
                let bits = self.u64()?;

                float_value(f64::from_bits(bits), bits)
            }
            _ => anyhow::bail!("unknown GGUF metadata type {kind}"),
        })
    }
}

fn float_value(value: f64, bits: u64) -> Value {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or_else(|| json!({"nonfinite_bits": bits}))
}
