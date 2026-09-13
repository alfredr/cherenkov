use super::*;
use serde_json::json;
use std::{fs, path::Path};

mod gguf;
mod packed;
mod raw;

#[derive(Default)]
struct Fixture {
    header: serde_json::Map<String, serde_json::Value>,
    data: Vec<u8>,
}

impl Fixture {
    fn tensor(&mut self, name: &str, dtype: &str, shape: &[usize]) {
        let width = match dtype {
            "U32" | "F32" => 4,
            "I64" => 8,
            _ => 2,
        };
        let start = self.data.len();

        self.data
            .resize(start + shape.iter().product::<usize>() * width, 0);
        self.header.insert(
            name.into(),
            json!({
                "dtype": dtype, "shape": shape, "data_offsets": [start, self.data.len()]
            }),
        );
    }

    fn affine(&mut self, prefix: &str, shape: &[usize], group: usize, bits: usize) {
        let mut packed = shape.to_vec();
        *packed.last_mut().unwrap() /= 32 / bits;

        self.tensor(&format!("{prefix}.weight"), "U32", &packed);

        *packed.last_mut().unwrap() = shape.last().unwrap() / group;

        self.tensor(&format!("{prefix}.scales"), "BF16", &packed);
        self.tensor(&format!("{prefix}.biases"), "BF16", &packed);
    }

    fn write(&self, dir: &Path, config: serde_json::Value) {
        crate::test_support::write_safetensors(
            &dir.join("model.safetensors"),
            &self.header,
            &self.data,
        );
        fs::write(
            dir.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    }
}

fn mlx_config(bits: usize) -> serde_json::Value {
    json!({"model_type": "qwen4_exp", "quantization": {"bits": bits, "group_size": 64, "mode": "affine"}})
}

#[test]
fn strided_views_validate_extent_and_overflow() {
    let mut view = StoredTensor {
        dtype: Dtype::Bf16,
        shape: vec![2, 5],
        byte_strides: vec![100, 2],
        data: DataSpan {
            object: ObjectId(0),
            offset: 80,
            length: 110,
        },
    };

    view.validate().unwrap();

    view.data.length = 109;

    assert!(view.validate().is_err());

    view.data.length = 110;
    view.byte_strides[0] = u64::MAX;

    assert!(view.validate().is_err());
}
