use super::*;
use crate::test_support::mlx_checkpoint::checkpoint;

fn packed_checkpoint() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();

    checkpoint(dir.path());
    prepare(dir.path(), None, &[4]).unwrap();

    dir
}

#[test]
fn manifest_rejects_inconsistent_expert_dimensions() {
    let dir = packed_checkpoint();
    let packed = dir.path().join("packed");
    let manifest = Manifest::load(&packed).unwrap();
    let mutations: [fn(&mut ExpertLayout); 5] = [
        |layout| {
            layout.layer_prefixes.pop();
        },
        |layout| {
            layout.layer_prefixes.push("extra".into());
        },
        |layout| layout.layers = 0,
        |layout| layout.experts = 0,
        |layout| {
            layout.layers = 2;
            layout.layer_prefixes = vec!["first".into(), "second".into()];
            layout.experts = usize::MAX;
        },
    ];

    for mutate in mutations {
        let mut invalid = manifest.clone();

        mutate(&mut invalid.experts);
        std::fs::write(
            packed.join("manifest.json"),
            serde_json::to_vec(&invalid).unwrap(),
        )
        .unwrap();
        assert!(Manifest::load(&packed).is_err());
    }
}

#[test]
fn fresh_checkpoint_builds_base_and_both_targets_at_custom_output() {
    let dir = tempfile::tempdir().unwrap();

    checkpoint(dir.path());

    let template = "{{ messages[0].content }}";

    std::fs::write(dir.path().join("chat_template.jinja"), template).unwrap();
    std::fs::write(dir.path().join("tokenizer_config.json"), "{}").unwrap();

    let out = dir.path().join("custom/nested/store");

    prepare(dir.path(), Some(&out), &[2, 3]).unwrap();

    for name in [
        "manifest.json",
        "experts.bin",
        "experts2.bin",
        "experts3.bin",
        "manifest2.json",
        "manifest3.json",
        "dense.bin",
        "ngram.bin",
        "config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "tokenizer_config.json",
    ] {
        assert!(out.join(name).is_file(), "{name}");
    }

    assert!(!dir.path().join("packed").exists());
    assert_eq!(
        std::fs::read_to_string(out.join("chat_template.jinja")).unwrap(),
        template
    );

    let base = std::fs::read(out.join("experts.bin")).unwrap();

    std::fs::remove_file(out.join("chat_template.jinja")).unwrap();
    prepare(dir.path(), Some(&out), &[4, 2, 3]).unwrap();
    assert_eq!(base, std::fs::read(out.join("experts.bin")).unwrap());
    assert_eq!(
        std::fs::read_to_string(out.join("chat_template.jinja")).unwrap(),
        template
    );
}

#[test]
fn default_pack_can_be_extended_from_the_packed_directory() {
    let dir = packed_checkpoint();
    let out = dir.path().join("packed");

    assert!(!out.join("experts2.bin").exists());
    assert!(!out.join("experts3.bin").exists());
    prepare(&out, None, &[3]).unwrap();
    assert!(out.join("experts3.bin").is_file());
    assert!(!out.join("experts2.bin").exists());
    assert!(!out.join("packed").exists());

    let absent = dir.path().join("another");

    assert!(prepare(&out, Some(&absent), &[2]).is_err());
    assert!(!absent.exists());
}

#[test]
fn output_from_another_checkpoint_is_rejected() {
    let dir = packed_checkpoint();

    let other = tempfile::tempdir().unwrap();
    let out = dir.path().join("packed");
    let error = prepare(other.path(), Some(&out), &[2]).unwrap_err();

    assert!(error.to_string().contains("different source model"));
    assert!(!out.join("experts2.bin").exists());
}

#[test]
fn relocated_model_reuses_its_local_base_store() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");

    std::fs::create_dir(&source).unwrap();
    checkpoint(&source);
    prepare(&source, None, &[4]).unwrap();

    let relocated = dir.path().join("relocated");

    std::fs::rename(source, &relocated).unwrap();
    prepare(&relocated, None, &[2]).unwrap();
    assert!(relocated.join("packed/experts2.bin").is_file());
}

#[test]
fn invalid_selection_does_not_create_a_base_store() {
    let dir = tempfile::tempdir().unwrap();

    for targets in [vec![], vec![2, 1], vec![5]] {
        assert!(prepare(dir.path(), None, &targets).is_err());
    }

    assert!(!dir.path().join("packed").exists());
}

#[test]
fn mlx_tensor_bytes_are_preserved_in_every_store() {
    let dir = packed_checkpoint();

    let source = Source::load(dir.path()).unwrap();
    let out = dir.path().join("packed");
    let manifest = Manifest::load(&out).unwrap();
    let dense = std::fs::read(out.join("dense.bin")).unwrap();

    for tensor in &manifest.dense {
        let bytes = source.bytes(source.tensor(&tensor.name).unwrap()).unwrap();

        assert_eq!(
            &dense[tensor.offset as usize..(tensor.offset + tensor.nbytes) as usize],
            bytes
        );
    }

    let experts = std::fs::read(out.join("experts.bin")).unwrap();
    let layout = &manifest.experts;
    let prefix = &layout.layer_prefixes[0];

    for (projection, offsets) in [
        ("gate_proj", [layout.gate_w, layout.gate_s, layout.gate_b]),
        ("up_proj", [layout.up_w, layout.up_s, layout.up_b]),
        ("down_proj", [layout.down_w, layout.down_s, layout.down_b]),
    ] {
        for (part, offset) in ["weight", "scales", "biases"].into_iter().zip(offsets) {
            let bytes = source
                .bytes(
                    source
                        .tensor(&format!("{prefix}.{projection}.{part}"))
                        .unwrap(),
                )
                .unwrap();
            let per_expert = bytes.len() / layout.experts;

            for expert in 0..layout.experts {
                let start = expert * layout.record_stride as usize + offset as usize;

                assert_eq!(
                    &experts[start..start + per_expert],
                    &bytes[expert * per_expert..(expert + 1) * per_expert]
                );
            }
        }
    }

    let ngram = std::fs::read(out.join("ngram.bin")).unwrap();
    let mut expected = Vec::new();

    for row in 0..2 {
        for part in ["weight", "scales", "biases"] {
            let name = format!("language_model.model.ple.ngram_embedding.shards.0.{part}");
            let bytes = source.bytes(source.tensor(&name).unwrap()).unwrap();
            let row_bytes = bytes.len() / 2;

            expected.extend_from_slice(&bytes[row * row_bytes..(row + 1) * row_bytes]);
        }
    }

    assert_eq!(ngram, expected);
}
