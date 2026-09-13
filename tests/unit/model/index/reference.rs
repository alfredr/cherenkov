use super::*;
use crate::model::index::reference::find;

/// Register an HF identity from fixture metadata without contacting a remote hub.
fn register_hf(index: &ModelIndex, repo: &str, revision: &str, endpoint: &str) -> String {
    let path = checkpoint(&index.paths.scratch);
    let description = Checkpoint::open(&path).unwrap().description;

    index
        .register(
            Source::HuggingFace {
                repo: repo.into(),
                revision: revision.into(),
                endpoint: endpoint.into(),
            },
            description,
            None,
            None,
        )
        .unwrap()
}

#[test]
fn unnamed_hf_models_report_source_references_and_accept_short_revisions() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());
    let revision = "abcdef01".repeat(5);
    let id = register_hf(
        &index,
        "example/model-q4",
        &revision,
        "https://huggingface.co",
    );
    let model = index.show(&format!("model:{id}")).unwrap();

    assert_eq!(model.summary.reference, "hf://example/model-q4");
    assert!(model.summary.name.is_none());
    assert_eq!(index.resolve(&model.summary.reference).unwrap().id, id);
    assert_eq!(
        index
            .resolve("model:hf:example/model-q4:abcdef01")
            .unwrap()
            .id,
        id
    );
    assert!(index.resolve("model:hf:example/model-q4:abcdef0").is_err());
    assert!(index.resolve("model:hf:example/model-q4:main").is_err());
    assert!(index.resolve("model:hf:example/model-q4:zzzzzzzz").is_err());

    let entry = index.resolve(&model.summary.reference).unwrap();

    index
        .register(entry.source, entry.description, Some("custom"), None)
        .unwrap();

    assert_eq!(
        index.show("model:custom").unwrap().summary.reference,
        "custom"
    );
    assert_eq!(index.resolve(&model.summary.reference).unwrap().id, id);
}

#[test]
fn short_revision_collisions_fail_without_removing_either_model() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());
    let first = format!("abcdef01{}", "a".repeat(32));
    let second = format!("abcdef01{}", "b".repeat(32));
    let a = register_hf(&index, "example/model", &first, "https://huggingface.co");
    let b = register_hf(&index, "example/model", &second, "https://huggingface.co");
    let short = "model:hf:example/model:abcdef01";

    assert!(
        index
            .resolve(short)
            .err()
            .unwrap()
            .to_string()
            .contains("ambiguous")
    );
    assert!(index.remove(short, false).is_err());
    assert_eq!(index.list().unwrap().len(), 2);
    assert_eq!(
        index
            .resolve("model:hf:example/model:abcdef01a")
            .unwrap()
            .id,
        a
    );
    assert_eq!(
        index
            .resolve(&format!("model:hf:example/model:{second}"))
            .unwrap()
            .id,
        b
    );
}

#[test]
fn source_references_distinguish_repository_owners_and_quantizations() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());
    let revision = "a".repeat(40);

    for repo in ["first/model-q4", "second/model-q4", "first/model-q8"] {
        let id = register_hf(&index, repo, &revision, "https://huggingface.co");

        assert_eq!(
            index
                .resolve(&format!("model:hf:{repo}:aaaaaaaa"))
                .unwrap()
                .id,
            id
        );
    }
}

#[test]
fn identical_repository_revisions_on_different_hubs_fall_back_to_ids() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());
    let revision = "a".repeat(40);

    register_hf(&index, "example/model", &revision, "https://huggingface.co");
    register_hf(&index, "example/model", &revision, "https://hub.example");

    for model in index.list().unwrap() {
        assert_eq!(model.reference, model.id);
        assert_eq!(index.resolve(&model.reference).unwrap().id, model.id);
    }
}

#[test]
fn local_references_preserve_paths_and_track_changed_sources() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());
    let path = checkpoint(&dir.path().join("local:model with spaces"));
    let first = index.add(path.to_str().unwrap(), None, None, None).unwrap();
    let entry = index.resolve(&first.summary.reference).unwrap();
    let Source::Local { path, fingerprint } = entry.source else {
        panic!("local source expected");
    };
    let short = format!("model:local:{}:{}", path.display(), &fingerprint[..8]);

    assert_eq!(first.summary.reference, path.to_string_lossy());
    assert_eq!(index.resolve(&short).unwrap().id, first.summary.id);

    std::fs::write(path.join("config.json"), r#"{"model_type":"changed"}"#).unwrap();

    let second = index.add(path.to_str().unwrap(), None, None, None).unwrap();

    assert_ne!(first.summary.reference, second.summary.reference);
    assert_eq!(index.resolve(&short).unwrap().id, first.summary.id);
    assert_eq!(
        index.resolve(&second.summary.reference).unwrap().id,
        second.summary.id
    );
    assert!(!is_reference(Path::new(
        "./model:hf:example/model:aaaaaaaa"
    )));
}

#[test]
fn known_hf_sources_reuse_pins_without_remote_inspection() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());
    let id = register_hf(
        &index,
        "Qwen/Tiny",
        &"abcdef01".repeat(5),
        "https://unused.invalid",
    );

    for (reference, revision) in [
        ("hf://Qwen/Tiny", None),
        ("hf://Qwen/Tiny@abcdef01", None),
        ("hf://Qwen/Tiny", Some("abcdef01")),
    ] {
        assert_eq!(
            index
                .add(reference, revision, None, None)
                .unwrap()
                .summary
                .id,
            id
        );
    }

    assert!(index.resolve("hf://Other/Tiny").is_err());
    assert!(index.add("unknown-alias", None, None, None).is_err());
    assert!(
        index
            .add("hf://Qwen/Tiny@abcdef01", Some("main"), None, None)
            .is_err()
    );
}

#[test]
fn invalid_revision_options_fail_before_registration() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());

    for revision in [
        "",
        "@main",
        "user:pass@host",
        "main?query",
        "main#fragment",
        "%61bcdef01",
        "a b",
        "a\nb",
        ".",
        "..",
        "/branch",
        "branch/",
        "feature/../branch",
    ] {
        let error = index
            .add("hf://Example/Tiny", Some(revision), None, None)
            .err()
            .unwrap();

        assert_eq!(error.to_string(), "invalid model revision", "{revision:?}");
    }

    assert!(!index.paths.data.join("index.json").exists());
}

#[test]
fn mutable_revisions_do_not_select_an_existing_pin() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());

    register_hf(
        &index,
        "Example/Tiny",
        &"a".repeat(40),
        "https://unused.invalid",
    );

    let catalog = index.lock().unwrap();

    for revision in ["main", "feature/branch"] {
        let reference = format!("hf://Example/Tiny@{revision}");
        let selector = selector::parse(Path::new(&reference)).unwrap();

        assert!(
            find(&catalog.value, &selector, &reference)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn select_preserves_lookup_errors() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());

    for tail in ["a", "b"] {
        let revision = format!("abcdef01{}", tail.repeat(32));

        register_hf(&index, "Example/Tiny", &revision, "https://unused.invalid");
    }

    for reference in [
        "hf://Example/Tiny",
        "hf://Example/Tiny@abcdef01",
        "model:hf:Example/Tiny:abcdef01",
        "missing",
        "model:missing",
    ] {
        let lookup = index.resolve(reference).err().unwrap().to_string();
        let selection = index
            .add(reference, None, None, None)
            .err()
            .unwrap()
            .to_string();

        assert_eq!(selection, lookup, "{reference}");
    }

    let error = index
        .add("hf://Example/Tiny", Some("abcdef01"), None, None)
        .err()
        .unwrap();

    assert_eq!(
        error.to_string(),
        "model reference \"hf://Example/Tiny@abcdef01\" is ambiguous; append @revision or use an alias or model ID"
    );
}

#[test]
fn anchoring_parses_before_deciding_whether_to_join_a_path() {
    let base = Path::new("/config");

    for reference in [
        "tiny",
        "hf://Qwen/Tiny",
        "disk://models/Qwen/Tiny",
        "model:tiny",
    ] {
        assert_eq!(
            anchor_selector(Path::new(reference), base).unwrap(),
            Path::new(reference)
        );
    }

    assert_eq!(
        anchor_selector(Path::new("./tiny"), base).unwrap(),
        base.join("./tiny")
    );
    assert!(anchor_selector(Path::new("unsupported://owner/repo"), base).is_err());
}

#[test]
fn prepare_and_server_defaults_select_the_same_offline_entry() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());
    let id = register_hf(
        &index,
        crate::storage::DEFAULT_REPO,
        crate::storage::DEFAULT_REVISION,
        "https://unused.invalid",
    );
    let config = crate::config::Config::default();
    let source = PathBuf::from(crate::storage::default_model_reference());

    assert_eq!(config.model_dir().unwrap(), source);
    assert_eq!(
        index
            .select(&source, ResolveOptions::default())
            .unwrap()
            .summary
            .id,
        id
    );
    assert_eq!(
        index
            .select(&config.model_dir().unwrap(), ResolveOptions::default())
            .unwrap()
            .summary
            .id,
        id
    );
    assert!(!index.paths.default_model().exists());
}
