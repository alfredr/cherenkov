use super::*;

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
fn gguf_q4k_is_not_mlx_affine_q4() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");

    fs::write(&path, fixture(12, 256, 160)).unwrap(); // 144-byte block, then padding

    let checkpoint = Checkpoint::open(&path).unwrap();
    let tensor = &checkpoint.description.tensors[0];

    assert_eq!(tensor.shape(), Some([1, 256].as_slice()));
    assert!(matches!(
        tensor.encoding,
        TensorEncoding::Ggml {
            encoding: GgmlEncoding::Q4K,
            ..
        }
    ));
    assert_eq!(
        checkpoint
            .map(tensor.encoding.data()[0])
            .unwrap()
            .as_ref()
            .len(),
        144
    );
    assert!(
        matches!(checkpoint.description.preparation(), Preparation::Unsupported { reasons } if matches!(reasons.as_slice(), [CompatibilityIssue::GgufConversion]))
    );
}
