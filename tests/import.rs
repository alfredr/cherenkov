//! Optional end-to-end checks against the Git LFS checkpoint.

use cherenkov::model::{Checkpoint, Preparation, TensorEncoding, TensorRole};
use cherenkov::options::{Options, PoolBudget};
use cherenkov::qwen4_exp::{cpu::CpuModel, gpu::Gpu, pack, packed::Packed};
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4_exp");
    let size = std::fs::metadata(path.join("model.safetensors")).map_or(0, |m| m.len());

    assert_eq!(
        size, 327_820_608,
        "fetch the fixture with `mise exec -- git lfs pull`"
    );

    path
}

fn assert_logits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual.iter().all(|v| v.is_finite()),
        "non-finite GPU logits"
    );
    assert!(
        expected.iter().all(|v| v.is_finite()),
        "non-finite reference logits"
    );

    let scale = expected.iter().map(|v| v.abs()).fold(1.0_f32, f32::max);
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    assert!(error < scale * 0.02, "logit error {error}, scale {scale}");
}

#[test]
#[ignore = "requires the Git LFS fixture and Metal"]
fn bf16_fixture_prefill_and_decode_match_cpu() {
    let source = fixture();
    let dir = tempfile::tempdir().unwrap();

    pack::prepare(&source, Some(dir.path()), &[4, 3, 2]).unwrap();

    let packed = Packed::open(dir.path()).unwrap();

    assert_eq!(packed.cfg.mtp_num_hidden_layers, 0);
    assert_eq!(packed.manifest.ngram.group, 16);
    assert!(dir.path().join("experts2.bin").exists());
    assert!(dir.path().join("experts3.bin").exists());

    let cpu = CpuModel::load(&packed).unwrap();
    let mut state = cpu.new_state();
    let options = Options {
        drafts: 0,
        pool_gb: PoolBudget::Gb(0.25),
        ..Options::default()
    };
    let mut gpu = Gpu::load(&packed, 64, &options).unwrap();
    let prompt: Vec<_> = (128..144).collect();
    let (mut next, _) = gpu.prefill_chunk(&prompt, None, true).unwrap();

    for (row, &token) in prompt.iter().enumerate() {
        assert_logits(
            gpu.pf_logits_row(row),
            &cpu.forward_token(token, &mut state).unwrap(),
        );
    }

    gpu.prefill_release();

    for _ in 0..8 {
        let reference = cpu.forward_token(next, &mut state).unwrap();
        next = gpu.step(next).unwrap();

        assert_logits(gpu.logits_row(0), &reference);

        let expected = reference
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;

        assert_eq!(next as usize, expected);
    }

    // The ordinary CLI must disable MTP and ignore tokenizer training padding.
    let index_root = tempfile::tempdir().unwrap();
    let output = generate_fixture(index_root.path(), dir.path());

    assert_fixture_output(output);
}

#[test]
#[ignore = "requires the Git LFS fixture"]
fn small_moe_native_weights_have_typed_shapes_and_retained_mappings() {
    let source = fixture();
    let checkpoint = Checkpoint::open(&source).unwrap();

    assert!(matches!(
        checkpoint.description.preparation(),
        Preparation::Required { .. }
    ));
    assert_eq!(checkpoint.description.tensors.len(), 271);

    let expert = checkpoint
        .description
        .tensors
        .iter()
        .find(|t| t.name == "model.language_model.layers.0.mlp.experts.gate_up_proj")
        .unwrap();

    assert_eq!(expert.role, TensorRole::Expert);
    assert_eq!(expert.shape(), Some([8, 512, 256].as_slice()));

    let bytes = checkpoint.map(expert.encoding.data()[0]).unwrap();

    assert_eq!(bytes.as_ref().len(), 8 * 512 * 256 * 2);
    drop(checkpoint);
    assert_eq!(bytes.as_ref().len(), 8 * 512 * 256 * 2);
}

#[test]
#[ignore = "requires the Git LFS fixture"]
fn small_moe_prepared_weights_have_strided_views() {
    let source = fixture();

    let dir = tempfile::tempdir().unwrap();

    pack::prepare(&source, Some(dir.path()), &[4]).unwrap();

    let checkpoint = Checkpoint::open(dir.path()).unwrap();

    assert!(matches!(
        checkpoint.description.preparation(),
        Preparation::Direct
    ));

    let experts: Vec<_> = checkpoint
        .description
        .tensors
        .iter()
        .filter(|t| t.role == TensorRole::Expert)
        .collect();

    assert_eq!(experts.len(), 12);
    assert!(
        experts
            .iter()
            .all(|t| matches!(t.encoding, TensorEncoding::Affine { bits: 4, .. }))
    );
    assert_eq!(
        checkpoint.description.ngram.as_ref().unwrap().shards.len(),
        1
    );

    for tensor in &checkpoint.description.tensors {
        tensor.validate().unwrap();

        for span in tensor.encoding.data() {
            checkpoint.map(span).unwrap();
        }
    }
}

fn pack_index(index: &cherenkov::model::index::ModelIndex, experts: &[u32], repack: bool) {
    index
        .pack(
            "model:tiny",
            cherenkov::model::index::PackOptions {
                output: None,
                experts,
                keep_source: false,
                token: None,
                repack,
            },
        )
        .unwrap();
}

#[test]
#[ignore = "requires the Git LFS fixture and Metal"]
fn indexed_model_survives_source_removal_and_variant_replacement() {
    use cherenkov::{model::index::ModelIndex, storage::Paths};

    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");

    copy_fixture(&source);

    let root = dir.path().join("index");
    let index = ModelIndex::new(Paths::new(Some(&root)).unwrap());

    index
        .add(source.to_str().unwrap(), None, Some("tiny"), None)
        .unwrap();
    pack_index(&index, &[4, 2], false);

    let old = index.acquire_prepared("model:tiny").unwrap();
    let old_q2 = std::fs::read(old.path.join("experts2.bin")).unwrap();

    std::fs::remove_dir_all(&source).unwrap();
    pack_index(&index, &[3, 2], true);

    let current = index.acquire_prepared("model:tiny").unwrap();
    let details = index.show("model:tiny").unwrap();

    assert_ne!(old.path, current.path);
    assert_eq!(details.summary.precisions, [4, 3, 2]);
    assert!(!details.summary.source_local);
    assert!(matches!(details.preparation, Preparation::Direct));
    assert_eq!(
        std::fs::read(old.path.join("experts2.bin")).unwrap(),
        old_q2
    );
    assert!(!old.path.join("experts3.bin").exists());
    assert_eq!(index.gc(false).unwrap().leased.len(), 1);

    let output = generate_fixture(&root, "model:tiny");

    assert_fixture_output(output);
    drop(old);
    assert_eq!(index.gc(false).unwrap().artifacts.len(), 1);
    index.remove("model:tiny", true).unwrap();
    index.remove("model:tiny", false).unwrap();
    assert_eq!(index.gc(false).unwrap().leased.len(), 1);
    drop(current);
    assert_eq!(index.gc(false).unwrap().artifacts.len(), 1);
}

fn copy_fixture(source: &Path) {
    std::fs::create_dir(source).unwrap();

    for entry in std::fs::read_dir(fixture()).unwrap() {
        let entry = entry.unwrap();

        if entry.path().is_file() {
            std::fs::hard_link(entry.path(), source.join(entry.file_name())).unwrap();
        }
    }
}

/// Generate the same short answer for path, alias, and legacy-selector checks.
fn generate_fixture(root: &Path, model: impl AsRef<std::ffi::OsStr>) -> std::process::Output {
    let mut args = vec![model.as_ref().to_owned()];

    args.extend(
        [
            "Hello",
            "--raw",
            "--max-tokens",
            "4",
            "--max-ctx",
            "64",
            "--pool-gb",
            "0.25",
        ]
        .map(std::ffi::OsString::from),
    );

    fixture_command(root, &args)
}

fn assert_fixture_output(output: std::process::Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        " my name is In\n"
    );
}

#[test]
#[ignore = "requires the Git LFS fixture and Metal"]
fn indexed_legacy_packed_directory_uses_parent_metadata() {
    use cherenkov::{model::index::ModelIndex, storage::Paths};

    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");

    copy_fixture(&source);
    pack::prepare(&source, None, &[4]).unwrap();

    // Keep the converted runtime metadata, but place it beside the source shards
    // as older stores did. Renaming also detaches the source fixture's hard links.
    for name in ["config.json", "tokenizer.json"] {
        std::fs::rename(source.join("packed").join(name), source.join(name)).unwrap();
    }

    let root = dir.path().join("index");
    let index = ModelIndex::new(Paths::new(Some(&root)).unwrap());

    index
        .add(source.to_str().unwrap(), None, Some("legacy"), None)
        .unwrap();

    let output = generate_fixture(&root, "model:legacy");

    assert_fixture_output(output);
    assert!(!source.join("packed/config.json").exists());
    assert!(!source.join("packed/tokenizer.json").exists());
    index
        .pack(
            "legacy",
            cherenkov::model::index::PackOptions {
                output: None,
                experts: &[4, 2],
                keep_source: false,
                token: None,
                repack: false,
            },
        )
        .unwrap();

    let current = index.acquire_prepared("legacy").unwrap();

    assert!(Packed::open(&current.path).is_ok());
    assert!(current.path.join("tokenizer.json").is_file());
    assert!(!source.join("packed/config.json").exists());
}

#[test]
#[ignore = "requires the Git LFS fixture"]
fn indexed_export_remains_external_after_collection() {
    use cherenkov::{
        model::index::{ModelIndex, PackOptions},
        storage::Paths,
    };

    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("index");
    let output = dir.path().join("export");
    let index = ModelIndex::new(Paths::new(Some(&root)).unwrap());

    index
        .add(fixture().to_str().unwrap(), None, Some("tiny"), None)
        .unwrap();

    let details = index
        .pack(
            "model:tiny",
            PackOptions {
                output: Some(&output),
                experts: &[4],
                keep_source: false,
                token: None,
                repack: false,
            },
        )
        .unwrap();

    assert!(!details.artifacts[0].owned);
    assert_eq!(details.summary.owned_bytes, 0);
    assert!(details.summary.external_bytes > 0);
    index.remove("model:tiny", false).unwrap();
    index.gc(false).unwrap();
    assert!(root.join("artifacts").read_dir().unwrap().next().is_none());
    assert!(Packed::open(&output).is_ok());
}

/// Run a fixture command against an isolated index and include stderr on failure.
fn fixture_command(root: &Path, args: &[impl AsRef<std::ffi::OsStr>]) -> std::process::Output {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_cherenkov"))
        .args(args)
        .arg("--root")
        .arg(root)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    output
}

#[test]
#[ignore = "requires the Git LFS fixture and Metal"]
fn prepare_from_a_named_store_runs_by_alias() {
    let dir = tempfile::tempdir().unwrap();
    let stores = dir.path().join("models");

    std::fs::create_dir_all(stores.join("Example")).unwrap();
    copy_fixture(&stores.join("Example/Tiny"));

    let root = dir.path().join("index");

    fixture_command(&root, &["store", "add", "models", stores.to_str().unwrap()]);
    fixture_command(
        &root,
        &[
            "prepare",
            "disk://models/Example/Tiny",
            "--name",
            "tiny",
            "--experts",
            "4,3,2",
        ],
    );

    let output = fixture_command(&root, &["inspect", "tiny", "--json"]);
    let inspected: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();

    assert_eq!(inspected["index"]["reference"], "tiny");
    assert_eq!(
        inspected["prepared_precisions"],
        serde_json::json!([4, 3, 2])
    );

    let output = generate_fixture(&root, "tiny");

    assert_fixture_output(output);
}
