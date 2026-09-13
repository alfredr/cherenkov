//! Small affine checkpoint for packing and model-index tests; no GPU or download.
use serde_json::json;
use std::path::Path;

pub fn checkpoint(dir: &Path) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut tensor = |name: String, dtype: &str, shape: &[usize], width: usize| {
        let start = data.len();
        let len = shape.iter().product::<usize>() * width;

        let bytes: Vec<u8> = if dtype == "I64" {
            std::iter::repeat_n(1_i64.to_le_bytes(), len / 8)
                .flatten()
                .collect()
        } else {
            (0..len).map(|i| (i * 37 + 11) as u8).collect()
        };

        data.extend(bytes);

        header.insert(
            name,
            json!({
                "dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]
            }),
        );
    };

    for projection in ["gate_proj", "up_proj", "down_proj"] {
        let prefix = format!("language_model.model.layers.0.mlp.switch_mlp.{projection}");

        tensor(format!("{prefix}.weight"), "U32", &[2, 64, 8], 4);
        tensor(format!("{prefix}.scales"), "BF16", &[2, 64, 1], 2);
        tensor(format!("{prefix}.biases"), "BF16", &[2, 64, 1], 2);
    }

    let ple = "language_model.model.ple";
    let ngram = format!("{ple}.ngram_embedding.shards.0");

    tensor(format!("{ngram}.weight"), "U32", &[2, 8], 4);
    tensor(format!("{ngram}.scales"), "BF16", &[2, 1], 2);
    tensor(format!("{ngram}.biases"), "BF16", &[2, 1], 2);

    for name in [
        "ngram_heads_offsets",
        "ngram_heads_vocab_sizes",
        "layer_multipliers",
    ] {
        tensor(format!("{ple}.{name}"), "I64", &[1], 8);
    }

    super::write_safetensors(&dir.join("model.safetensors"), &header, &data);

    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec(&json!({
            "model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 64}
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
}
