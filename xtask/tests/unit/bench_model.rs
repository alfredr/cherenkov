use super::*;

/// The CLI resolves both layouts to the directory containing the manifest.
fn inspection(path: &Path) -> Value {
    json!({"index": {
        "id": "0123456789abcdef0123456789abcdef", "precisions": [4, 2],
        "artifacts": [{"kind": "prepared", "available": true, "path": path}]
    }})
}

#[test]
fn legacy_signatures_require_a_matching_id_or_local_source() {
    let mut value = inspection(Path::new("/artifacts/prepared"));
    value["index"]["source"] = json!({"kind": "local", "path": "/models/original"});
    let model = Model::from_inspection(&value).unwrap();
    let digest: String = Sha256::digest(b"/models/original")
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();

    for mut signature in [
        json!({"model": model.reference}),
        json!({"model": "/models/original"}),
        json!({"model": "<model>", "model_path_sha256": digest}),
    ] {
        signature["model_metadata_sha256"] = json!({"packed/manifest.json": "hash"});

        model.migrate_signature(&mut signature);

        assert_eq!(
            signature,
            json!({
                "model": "<model>", "model_id": model.reference,
                "model_metadata_sha256": {"manifest.json": "hash"},
            })
        );
    }

    for original in [
        json!({"model": "<model>"}),
        json!({"model": "/models/other"}),
        json!({"model": "<model>", "model_path_sha256": "wrong"}),
        json!({"model": "/models/original", "model_id": "different-model"}),
    ] {
        let mut signature = original.clone();

        model.migrate_signature(&mut signature);
        assert_eq!(signature, original);
    }
}

#[test]
fn flat_and_legacy_stores_produce_the_same_metadata_hashes() {
    let root = tempfile::tempdir().unwrap();
    let flat = root.path().join("artifact");
    let legacy = root.path().join("model");

    std::fs::create_dir(&flat).unwrap();
    std::fs::create_dir_all(legacy.join("packed")).unwrap();

    for name in ["config.json", "tokenizer.json"] {
        std::fs::write(flat.join(name), name).unwrap();
        std::fs::write(legacy.join(name), name).unwrap();
    }

    std::fs::write(flat.join("manifest.json"), "manifest").unwrap();
    std::fs::write(legacy.join("packed/manifest.json"), "manifest").unwrap();

    let a = Model::from_inspection(&inspection(&flat)).unwrap();
    let b = Model::from_inspection(&inspection(&legacy.join("packed"))).unwrap();

    assert_eq!(a.metadata().unwrap(), b.metadata().unwrap());
    assert_eq!(a.precisions, [4, 2]);
}

#[test]
fn missing_prepared_artifacts_are_rejected_before_benchmarking() {
    for value in [json!({}), json!({"index": {"id": "id", "artifacts": []}})] {
        assert!(Model::from_inspection(&value).is_err());
    }

    let mut value = inspection(Path::new("/missing"));
    value["index"]["artifacts"][0]["available"] = json!(false);

    assert!(Model::from_inspection(&value).is_err());
}

#[test]
fn inspection_forwards_the_selector_and_root_as_distinct_arguments() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let binary = dir.path().join("fake-cli");
    let output = inspection(dir.path());
    let script = format!(
        "#!/bin/sh\n[ \"$1\" = inspect ] && [ \"$2\" = 'disk://models/Example/Tiny' ] && [ \"$3\" = --json ] && [ \"$4\" = --root ] && [ \"$5\" = '/index with spaces' ] || exit 1\ncat <<'JSON'\n{output}\nJSON\n"
    );

    std::fs::write(&binary, script).unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();

    let model = Model::inspect(
        &binary,
        OsStr::new("disk://models/Example/Tiny"),
        Some(Path::new("/index with spaces")),
    )
    .unwrap();

    assert_eq!(model.directory, dir.path());
    assert_eq!(model.reference, "0123456789abcdef0123456789abcdef");
}
