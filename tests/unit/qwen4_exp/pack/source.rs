use super::*;
use serde_json::json;

struct Fixture {
    header: serde_json::Map<String, serde_json::Value>,
    bytes: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            header: Default::default(),
            bytes: Vec::new(),
        }
    }

    fn tensor(&mut self, name: &str, shape: &[usize], values: impl IntoIterator<Item = f32>) {
        let start = self.bytes.len();

        self.bytes.extend(
            values
                .into_iter()
                .flat_map(|v| bf16::from_f32(v).to_le_bytes()),
        );
        assert_eq!(
            self.bytes.len() - start,
            shape.iter().product::<usize>() * 2
        );
        self.header.insert(
            name.into(),
            json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, self.bytes.len()]}),
        );
    }

    fn source(&self) -> Source {
        let dir = tempfile::tempdir().unwrap();

        crate::test_support::write_safetensors(
            &dir.path().join("model.safetensors"),
            &self.header,
            &self.bytes,
        );

        let raw = ModelWeights::load_raw(dir.path()).unwrap();
        let mut source = Source {
            raw,
            tensors: HashMap::new(),
            config: None,
        };

        for (name, info) in source.raw.tensors.clone() {
            source.import(&name, &info).unwrap();
        }

        source
    }
}

fn read(source: &Source, name: &str, range: Range<usize>) -> Vec<u8> {
    let mut bytes = Vec::new();

    source
        .write(source.tensor(name).unwrap(), range, &mut bytes)
        .unwrap();

    bytes
}

#[test]
fn fused_experts_select_each_half_within_each_expert() {
    let mut fixture = Fixture::new();
    // Two experts, each with a gate then an up matrix. Distinct constants
    // identify the four blocks independently of the quantization algorithm.
    let values = [1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(|v| std::iter::repeat_n(v, 64 * 64));

    fixture.tensor(
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        &[2, 128, 64],
        values,
    );

    let source = fixture.source();

    for (projection, expected) in [("gate_proj", [1.0, 3.0]), ("up_proj", [2.0, 4.0])] {
        let name = format!("language_model.model.layers.0.mlp.switch_mlp.{projection}.biases");
        let bytes = read(&source, &name, 0..256);
        let actual: Vec<_> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| bf16::from_le_bytes(b).to_f32())
            .collect();

        assert_eq!(&actual[..64], &[expected[0]; 64]);
        assert_eq!(&actual[64..], &[expected[1]; 64]);
        assert_eq!(read(&source, &name, 128..256), bytes[128..]);
    }
}

#[test]
fn norm_folding_preserves_gated_norms_mtp_norms_routers_and_convolutions() {
    let mut fixture = Fixture::new();
    let layer = "model.language_model.layers.0";
    let names = [
        format!("{layer}.attn_hyper_connection.hc_norm.weight"),
        format!("{layer}.linear_attn.norm.weight"),
        "mtp.pre_fc_norm_hidden.weight".into(),
        format!("{layer}.mlp.gate.weight"),
        format!("{layer}.linear_attn.conv1d.weight"),
    ];
    let shapes: [&[usize]; 5] = [&[4], &[4], &[4], &[1, 4], &[1, 1, 4]];

    for (name, shape) in names.iter().zip(shapes) {
        fixture.tensor(name, shape, [0.0, 0.5, -0.5, 1.0]);
    }

    let source = fixture.source();

    for (index, name) in names.iter().enumerate() {
        let name = normalized_name(name).unwrap();
        let actual = read(&source, &name, 0..8);
        let values = if index == 0 {
            [1.0, 1.5, 0.5, 2.0]
        } else {
            [0.0, 0.5, -0.5, 1.0]
        };
        let expected: Vec<_> = values
            .into_iter()
            .flat_map(|v| bf16::from_f32(v).to_le_bytes())
            .collect();

        assert_eq!(actual, expected, "{name}");
    }
}

#[test]
fn ngram_groups_follow_row_width() {
    for (width, group) in [(16, 16), (160, 32), (256, 64)] {
        let mut fixture = Fixture::new();
        let name = "model.language_model.layers.0.ple.ngram_embedding.shard_0.weight";

        fixture.tensor(name, &[2, width], std::iter::repeat_n(1.0, 2 * width));

        let source = fixture.source();
        let prefix = "language_model.model.layers.0.ple.ngram_embedding.shard_0";

        assert_eq!(
            source.tensor(&format!("{prefix}.weight")).unwrap().shape,
            [2, width / 8]
        );
        assert_eq!(
            source.tensor(&format!("{prefix}.scales")).unwrap().shape,
            [2, width / group]
        );
    }
}

#[test]
fn unaligned_matrix_width_is_rejected() {
    let mut fixture = Fixture::new();

    fixture.tensor("lm_head.weight", &[1, 64], [1.0; 64]);

    let mut source = fixture.source();
    let mut info = source.raw.tensors["lm_head.weight"].clone();
    info.shape = vec![1, 32];

    assert!(
        source
            .import("model.language_model.embed_tokens.weight", &info)
            .is_err()
    );
}

fn model_config() -> (std::path::PathBuf, Qwen4ExpConfig) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4_exp");
    let cfg = Qwen4ExpConfig::load(&dir).unwrap();

    (dir, cfg)
}

#[test]
fn absent_mtp_is_disabled_only_in_the_output_configuration() {
    let (dir, cfg) = model_config();
    let original = std::fs::read(dir.join("config.json")).unwrap();
    let mut source = Fixture::new().source();

    source.configure(&dir, &cfg).unwrap();

    let config = source.config.unwrap();

    assert_eq!(cfg.mtp_num_hidden_layers, 1);
    assert_eq!(config["text_config"]["mtp_num_hidden_layers"], 0);
    assert_eq!(config["text_config"]["mtp"]["num_hidden_layers"], 0);
    assert_eq!(std::fs::read(dir.join("config.json")).unwrap(), original);
}

#[test]
fn existing_mtp_is_preserved_and_partial_mtp_fails() {
    let (dir, cfg) = model_config();
    let mut fixture = Fixture::new();

    fixture.tensor("mtp.fc_embedding.weight", &[64, 64], [1.0; 64 * 64]);

    let mut partial = fixture.source();

    assert!(partial.configure(&dir, &cfg).is_err());
    fixture.tensor("mtp.fc_hidden.weight", &[64, 64], [2.0; 64 * 64]);

    let mut complete = fixture.source();

    complete.configure(&dir, &cfg).unwrap();
    assert_eq!(
        complete.config.unwrap()["text_config"]["mtp_num_hidden_layers"],
        1
    );
}
