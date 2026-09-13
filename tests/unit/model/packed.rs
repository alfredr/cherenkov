use super::*;

#[test]
fn interleaved_ngram_components_have_independent_offsets_and_shared_row_stride() {
    for (dim, group, row_bytes) in [(160, 32, 100), (16, 16, 12)] {
        let dir = tempfile::tempdir().unwrap();
        let mut fixture = Fixture::default();

        for projection in ["gate_proj", "up_proj", "down_proj"] {
            fixture.affine(
                &format!("language_model.model.layers.0.mlp.switch_mlp.{projection}"),
                &[2, 64, 64],
                64,
                4,
            );
        }

        let prefix = "language_model.model.ple";

        fixture.affine(
            &format!("{prefix}.ngram_embedding.shards.0"),
            &[3, dim],
            group,
            4,
        );

        for suffix in [
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
            "layer_multipliers",
        ] {
            fixture.tensor(&format!("{prefix}.{suffix}"), "I64", &[1]);
        }

        fixture.write(dir.path(), mlx_config(4));
        crate::qwen4_exp::pack::prepare(dir.path(), None, &[4]).unwrap();

        let checkpoint = Checkpoint::open(&dir.path().join("packed")).unwrap();
        let table = checkpoint.description.ngram.as_ref().unwrap();
        let tensor = &checkpoint.description.tensors[table.shards[0].tensor.0];
        let TensorEncoding::Affine {
            codes,
            scales,
            offset: AffineOffset::Bias { tensor: biases },
            ..
        } = &tensor.encoding
        else {
            panic!("expected affine n-gram")
        };

        assert_eq!(tensor.shape(), Some([3, dim as u64].as_slice()));
        assert_eq!(codes.data.offset, 0);
        assert_eq!(scales.data.offset, (dim / 2) as u64);
        assert_eq!(biases.data.offset, (dim / 2 + dim / group * 2) as u64);

        for part in [codes, scales, biases] {
            assert_eq!(part.byte_strides[0], row_bytes);
            part.validate().unwrap();
            checkpoint.map(&part.data).unwrap();
        }

        assert!(matches!(
            checkpoint.description.preparation(),
            Preparation::Direct
        ));
    }
}
