use cherenkov_model_data::*;
use std::fs;

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend((value.len() as u64).to_le_bytes());
    bytes.extend(value.as_bytes());
}

fn fixture(code: u32, width: u64, size: usize) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();

    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(1_u64.to_le_bytes()); // tensors
    bytes.extend(1_u64.to_le_bytes()); // metadata
    string(&mut bytes, "general.architecture");
    bytes.extend(8_u32.to_le_bytes());
    string(&mut bytes, "qwen4_exp");
    string(&mut bytes, "weight");
    bytes.extend(2_u32.to_le_bytes());
    bytes.extend(width.to_le_bytes());
    bytes.extend(1_u64.to_le_bytes());
    bytes.extend(code.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.resize(bytes.len().next_multiple_of(32) + size, 0);

    bytes
}

#[test]
fn gguf_codecs_preserve_block_sizes_and_unknown_codes() {
    for (code, width, size, expected_len) in [(12, 256, 160, 144), (999, 64, 39, 39)] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");

        fs::write(&path, fixture(code, width, size)).unwrap();

        let checkpoint = Checkpoint::open(&path).unwrap();
        let tensor = &checkpoint.inventory.tensors[0];

        assert_eq!(tensor.shape(), Some([1, width].as_slice()));

        let TensorEncoding::Ggml { encoding, data } = &tensor.encoding else {
            panic!("GGUF encoding lost");
        };
        let actual_code = match encoding {
            GgmlEncoding::Q4K => 12,
            GgmlEncoding::Opaque { code } => *code,
            other => panic!("unexpected encoding {other:?}"),
        };

        assert_eq!(actual_code, code);
        assert_eq!(
            checkpoint.objects.view(data).unwrap().as_ref().len(),
            expected_len
        );
    }
}

#[test]
fn gguf_rejects_truncated_headers_blocks_and_bad_dimensions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.gguf");

    for bytes in [
        b"GGUF".to_vec(),
        fixture(12, 256, 143),
        fixture(12, 255, 144),
        fixture(0, u64::MAX, 4),
    ] {
        fs::write(&path, bytes).unwrap();
        assert!(Checkpoint::open(&path).is_err());
    }
}

#[test]
fn gguf_retains_typed_metadata_and_nonfinite_float_bits() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metadata.gguf");
    let mut bytes = b"GGUF".to_vec();

    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    string(&mut bytes, "custom.layers");
    bytes.extend(9_u32.to_le_bytes()); // array
    bytes.extend(5_u32.to_le_bytes()); // int32 elements
    bytes.extend(2_u64.to_le_bytes());
    bytes.extend((-3_i32).to_le_bytes());
    bytes.extend(7_i32.to_le_bytes());
    string(&mut bytes, "custom.nan");
    bytes.extend(6_u32.to_le_bytes()); // float32
    bytes.extend(0x7fc01234_u32.to_le_bytes());
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    fs::write(&path, bytes).unwrap();

    let checkpoint = Checkpoint::open(&path).unwrap();
    let metadata = &checkpoint.inventory.metadata;

    assert_eq!(metadata["custom.layers"]["type"], 9);
    assert_eq!(metadata["custom.layers"]["value"]["element_type"], 5);
    assert_eq!(
        metadata["custom.layers"]["value"]["values"],
        serde_json::json!([-3, 7])
    );
    assert_eq!(
        metadata["custom.nan"]["value"]["nonfinite_bits"],
        0x7fc01234_u32
    );
}
