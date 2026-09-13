use super::*;

/// Place a metadata-only checkpoint at a store's owner/repository path.
fn disk_checkpoint(root: &Path) -> PathBuf {
    let source = checkpoint(&root.join("Example"));
    let target = root.join("Example/Tiny");

    std::fs::rename(source, &target).unwrap();

    target
}

#[test]
fn named_store_resolution_reuses_path_identity_and_preserves_external_files() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(&dir.path().join("index"));
    let root = dir.path().join("models");
    let source = disk_checkpoint(&root);
    let store = index
        .add_disk_store("models", &root, DiskLayout::Directory)
        .unwrap();
    let selected = index
        .add("disk://models/Example/Tiny", None, None, None)
        .unwrap();
    let local = index
        .add(source.to_str().unwrap(), None, None, None)
        .unwrap();

    assert_eq!(selected.summary.id, local.summary.id);
    assert_eq!(selected.summary.reference, "disk://models/Example/Tiny");
    assert_eq!(index.disk_stores().unwrap()[0].id, store.id);
    assert!(
        index
            .add_disk_store("models", &root, DiskLayout::Directory)
            .is_err()
    );

    index
        .add("disk://models/Example/Tiny", None, Some("tiny"), None)
        .unwrap();
    index.set_disk_store_enabled("models", false).unwrap();

    assert!(index.resolve("disk://models/Example/Tiny").is_err());
    assert_eq!(index.resolve("tiny").unwrap().id, selected.summary.id);

    index.set_disk_store_enabled("models", true).unwrap();
    assert_eq!(
        index.resolve("disk://models/Example/Tiny").unwrap().id,
        selected.summary.id
    );
    index.remove_disk_store("models").unwrap();
    index.remove("tiny", false).unwrap();
    index.gc(false).unwrap();

    assert!(source.join("model.safetensors").is_file());
    assert!(index.disk_stores().unwrap().is_empty());
    assert_ne!(
        index
            .add_disk_store("models", &root, DiskLayout::Directory)
            .unwrap()
            .id,
        store.id
    );
}

#[test]
fn cache_layout_resolves_refs_and_rejects_ambiguous_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(&dir.path().join("index"));
    let cache = dir.path().join("cache");
    let repo = cache.join("models--Example--Tiny");
    let source = checkpoint(&repo.join("snapshots"));
    let commit = "abcdef01".repeat(5);
    let target = repo.join("snapshots").join(&commit);

    std::fs::rename(source, &target).unwrap();
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::write(repo.join("refs/main"), &commit).unwrap();
    index
        .add_disk_store("cache", &cache, DiskLayout::HfCache)
        .unwrap();

    let model = index
        .add("disk://cache/Example/Tiny@main", None, None, None)
        .unwrap();

    assert_eq!(model.summary.reference, "disk://cache/Example/Tiny");
    assert_eq!(
        index
            .resolve("disk://cache/Example/Tiny@abcdef01")
            .unwrap()
            .id,
        model.summary.id
    );
    assert_eq!(
        index.resolve("disk://cache/Example/Tiny").unwrap().id,
        model.summary.id
    );

    for revision in ["a", "abcdef0", "missing-branch", "bbbbbbbb"] {
        let reference = format!("disk://cache/Example/Tiny@{revision}");

        assert!(
            index.add(&reference, None, None, None).is_err(),
            "{reference}"
        );
        assert!(index.resolve(&reference).is_err(), "{reference}");
    }

    let reference = "disk://cache/Example/Tiny@abcdef01";

    assert_eq!(
        index.add(reference, None, None, None).unwrap().summary.id,
        model.summary.id
    );

    std::fs::remove_file(repo.join("refs/main")).unwrap();
    std::fs::create_dir(repo.join("snapshots").join("b".repeat(40))).unwrap();

    let store = index.disk_stores().unwrap().remove(0);

    assert!(store.locate("Example/Tiny", None).is_err());
}

#[test]
fn stores_reject_symlink_escapes() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(&dir.path().join("index"));
    let outside = checkpoint(&dir.path().join("outside"));
    let root = dir.path().join("models");

    std::fs::create_dir_all(root.join("Example")).unwrap();
    std::os::unix::fs::symlink(outside, root.join("Example/Tiny")).unwrap();
    index
        .add_disk_store("models", &root, DiskLayout::Directory)
        .unwrap();
    assert!(
        index
            .add("disk://models/Example/Tiny", None, None, None)
            .is_err()
    );
}

#[test]
fn indexed_disk_sources_remain_selectable_offline_but_respect_disabled_stores() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(&dir.path().join("index"));
    let root = dir.path().join("models");

    disk_checkpoint(&root);
    index
        .add_disk_store("models", &root, DiskLayout::Directory)
        .unwrap();

    let reference = "disk://models/Example/Tiny";
    let selected = index.add(reference, None, Some("tiny"), None).unwrap();
    let Source::Local { fingerprint, .. } = index.resolve("tiny").unwrap().source else {
        panic!("local source expected");
    };
    let qualified = format!("{reference}@{}", &fingerprint[..8]);

    std::fs::remove_dir_all(&root).unwrap();

    for reference in [reference, &qualified] {
        assert_eq!(
            index.add(reference, None, None, None).unwrap().summary.id,
            selected.summary.id
        );
    }

    index.set_disk_store_enabled("models", false).unwrap();

    for reference in [reference, &qualified] {
        let error = index.add(reference, None, None, None).err().unwrap();

        assert!(error.to_string().contains("disabled"));
    }

    assert_eq!(
        index.add("tiny", None, None, None).unwrap().summary.id,
        selected.summary.id
    );
}
