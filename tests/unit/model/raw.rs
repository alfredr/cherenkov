use super::*;

const EXPERT: &str = "language_model.model.layers.0.mlp.switch_mlp.gate_proj";

#[test]
fn preparation_distinguishes_q4_group_sizes_and_q8() {
    for (bits, group, expected) in [
        (4, 64, "required"),
        (4, 32, "group_size"),
        (8, 64, "encoding"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut fixture = Fixture::default();

        fixture.affine(EXPERT, &[2, 64, 64], group, bits);
        fixture.write(dir.path(), mlx_config(bits));

        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let result = serde_json::to_value(checkpoint.description.preparation()).unwrap();

        if expected == "required" {
            assert_eq!(result["status"], expected);
        } else {
            assert_eq!(result["status"], "unsupported");
            assert_eq!(result["reasons"][0]["kind"], expected);
        }

        let tensor = &checkpoint.description.tensors[0];

        assert_eq!(tensor.shape(), Some([2, 64, 64].as_slice()));
        assert!(
            matches!(tensor.encoding, TensorEncoding::Affine { bits: b, .. } if usize::from(b) == bits)
        );
    }
}

#[test]
fn undeclared_quantization_is_opaque_and_keeps_all_components() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::default();

    fixture.affine(EXPERT, &[2, 64, 64], 64, 4);
    fixture.write(dir.path(), json!({"model_type": "qwen4_exp"}));

    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let tensor = &checkpoint.description.tensors[0];

    assert!(matches!(tensor.encoding, TensorEncoding::Opaque { .. }));
    assert!(tensor.shape().is_none());
    assert_eq!(tensor.encoding.data().len(), 3);
    assert!(matches!(
        checkpoint.description.preparation(),
        Preparation::Unsupported { .. }
    ));

    let roundtrip: ModelDescription =
        serde_json::from_slice(&serde_json::to_vec(&checkpoint.description).unwrap()).unwrap();

    assert_eq!(roundtrip.tensors[0].encoding.data().len(), 3);
}

#[test]
fn native_bf16_has_conversion_steps_and_typed_roles() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::default();

    fixture.tensor(
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "BF16",
        &[2, 128, 64],
    );
    fixture.tensor(
        "model.language_model.layers.0.mlp.gate.weight",
        "BF16",
        &[2, 64],
    );
    fixture.tensor(
        "model.language_model.embed_tokens.weight",
        "BF16",
        &[64, 64],
    );
    fixture.write(dir.path(), json!({"model_type": "qwen4_exp"}));

    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let Preparation::Required { steps } = checkpoint.description.preparation() else {
        panic!("expected conversion")
    };

    assert!(steps.contains(&PreparationStep::QuantizeAffineQ4));
    assert!(
        checkpoint
            .description
            .tensors
            .iter()
            .any(|t| t.role == TensorRole::Router)
    );
    assert!(
        checkpoint
            .description
            .tensors
            .iter()
            .any(|t| t.role == TensorRole::Embedding)
    );
    assert_eq!(
        qwen::tensor_role("model.surprise.weight", 2),
        TensorRole::Opaque
    );
}

#[test]
fn ngram_shards_use_numeric_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::default();
    let prefix = "model.language_model.layers.0.ple.ple_embedding";

    for shard in 0..12 {
        fixture.tensor(
            &format!("{prefix}.ngram_embedding.shard_{shard}.weight"),
            "BF16",
            &[shard + 1, 16],
        );
    }

    for suffix in [
        "ngram_heads_offsets",
        "ngram_heads_vocab_sizes",
        "layer_multipliers",
    ] {
        fixture.tensor(&format!("{prefix}.{suffix}"), "I64", &[1]);
    }

    fixture.write(dir.path(), json!({"model_type": "qwen4_exp"}));

    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let table = checkpoint.description.ngram.as_ref().unwrap();

    assert_eq!(table.shards.len(), 12);
    assert_eq!(table.shards[10].first_row, 55);
    assert_eq!(table.shards[10].rows, 11);
    assert!(
        checkpoint.description.tensors[table.shards[10].tensor.0]
            .name
            .ends_with("shard_10.weight")
    );
}

#[test]
fn other_mlx_architectures_are_inspectable_without_claiming_execution_support() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::default();

    fixture.affine("model.layers.0.mlp.gate_proj", &[64, 64], 64, 4);

    let mut config = mlx_config(4);
    config["model_type"] = json!("qwen3_5");

    fixture.write(dir.path(), config);

    let checkpoint = Checkpoint::open(dir.path()).unwrap();

    assert!(matches!(
        checkpoint.description.format.conventions,
        CheckpointConventions::MlxAffine
    ));
    assert!(matches!(
        checkpoint.description.tensors[0].encoding,
        TensorEncoding::Affine { bits: 4, .. }
    ));
    assert_eq!(checkpoint.description.tensors[0].role, TensorRole::Opaque);
    assert!(
        matches!(checkpoint.description.preparation(), Preparation::Unsupported { reasons }
        if matches!(reasons.as_slice(), [CompatibilityIssue::Architecture]))
    );
}

#[test]
fn opening_a_source_does_not_silently_select_its_prepared_copy() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::default();

    fixture.tensor("w", "F32", &[2]);
    fixture.write(dir.path(), json!({"model_type":"unknown"}));

    let packed = dir.path().join("packed");

    fs::create_dir(&packed).unwrap();
    fs::write(packed.join("manifest.json"), "invalid manifest").unwrap();

    let checkpoint = Checkpoint::open(dir.path()).unwrap();

    assert!(matches!(
        checkpoint.description.format.container,
        ContainerFormat::Safetensors
    ));
    assert!(Checkpoint::open(&packed).is_err());
}
