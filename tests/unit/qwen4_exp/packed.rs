use super::*;
use crate::test_support::mlx_checkpoint::checkpoint;

const PREFIX: &str = "language_model.model.ple";
const METADATA: [(&str, i64); 3] = [
    ("layer_multipliers", i64::MIN + 3),
    ("ngram_heads_offsets", 1 << 40),
    ("ngram_heads_vocab_sizes", 97),
];

fn packed_checkpoint() -> (tempfile::TempDir, Packed) {
    let dir = tempfile::tempdir().unwrap();

    checkpoint(dir.path());
    crate::qwen4_exp::pack::prepare(dir.path(), None, &[4]).unwrap();

    let path = dir.path().join("packed");
    let manifest = Manifest::load(&path).unwrap();
    let mut dense = std::fs::read(path.join("dense.bin")).unwrap();

    for (suffix, value) in METADATA {
        let entry = manifest.dense(&format!("{PREFIX}.{suffix}")).unwrap();
        let offset = entry.offset as usize;

        dense[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    std::fs::write(path.join("dense.bin"), dense).unwrap();
    std::fs::write(
        path.join("config.json"),
        include_str!("../../fixtures/qwen4_exp/config.json"),
    )
    .unwrap();

    let packed = Packed::open(&path).unwrap();

    (dir, packed)
}

#[test]
fn ngram_metadata_reads_dense_values_without_using_manifest_copies() {
    let (_dir, packed) = packed_checkpoint();
    let metadata = packed.ngram_metadata(PREFIX).unwrap();

    assert_eq!(metadata.multipliers, [i64::MIN + 3]);
    assert_eq!(metadata.head_offsets, [1 << 40]);
    assert_eq!(metadata.head_sizes, [97]);
    assert_ne!(metadata.head_offsets, packed.manifest.ngram.head_offsets);
}

#[test]
fn ngram_metadata_rejects_missing_tensors_and_wrong_dtypes() {
    let (_dir, mut packed) = packed_checkpoint();
    let manifest = packed.manifest.clone();

    for (suffix, _) in METADATA {
        let name = format!("{PREFIX}.{suffix}");

        packed.manifest = manifest.clone();

        packed.manifest.dense.retain(|entry| entry.name != name);

        let error = packed.ngram_metadata(PREFIX).unwrap_err();

        assert!(error.to_string().contains(&name), "{error}");
        assert!(error.to_string().contains("not in manifest"), "{error}");

        packed.manifest = manifest.clone();
        packed
            .manifest
            .dense
            .iter_mut()
            .find(|entry| entry.name == name)
            .unwrap()
            .dtype = "U32".into();

        let error = packed.ngram_metadata(PREFIX).unwrap_err();

        assert!(error.to_string().contains(&name), "{error}");
        assert!(error.to_string().contains("expected I64"), "{error}");
    }
}
