use super::*;
use serde_json::json;
use std::path::{Path, PathBuf};

fn index(dir: &Path) -> ModelIndex {
    ModelIndex::new(Paths {
        data: dir.join("data"),
        scratch: dir.join("scratch"),
        config: dir.join("config.toml"),
    })
}

fn checkpoint(dir: &Path) -> PathBuf {
    let path = dir.join("source");

    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(path.join("config.json"), r#"{"model_type":"example"}"#).unwrap();

    crate::test_support::write_safetensors(
        &path.join("model.safetensors"),
        &json!({"weight": {"dtype":"F32", "shape":[1], "data_offsets":[0,4]}}),
        &1.0_f32.to_le_bytes(),
    );

    path
}

/// Own the temporary index and source for registration and artifact-lifecycle tests.
struct Fixture {
    _dir: tempfile::TempDir,
    index: ModelIndex,
    source: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path());
        let source = checkpoint(dir.path());

        Self {
            _dir: dir,
            index,
            source,
        }
    }

    fn register(&self, name: &str) -> ModelDetails {
        self.index
            .add(self.source.to_str().unwrap(), None, Some(name), None)
            .unwrap()
    }

    fn registered(name: &str) -> Self {
        let fixture = Self::new();

        fixture.register(name);

        fixture
    }
}

#[test]
fn registration_is_idempotent_and_names_do_not_change_identity() {
    let fixture = Fixture::new();
    let index = &fixture.index;
    let path = &fixture.source;
    let first = index.add(path.to_str().unwrap(), None, None, None).unwrap();
    let named = fixture.register("example");

    assert_eq!(first.summary.id, named.summary.id);
    assert_eq!(index.resolve("model:example").unwrap().id, first.summary.id);
    assert_eq!(index.list().unwrap().len(), 1);
    assert_eq!(index.resolve("example").unwrap().id, first.summary.id);
    assert!(
        index
            .add(path.to_str().unwrap(), None, Some("other-name"), None)
            .is_err()
    );
}

#[test]
fn changed_local_sources_require_a_new_entry() {
    let fixture = Fixture::new();
    let index = &fixture.index;
    let path = &fixture.source;
    let first = fixture.register("example");

    std::fs::write(path.join("config.json"), r#"{"model_type":"different"}"#).unwrap();
    assert!(
        index
            .add(path.to_str().unwrap(), None, Some("example"), None)
            .is_err()
    );

    let second = index.add(path.to_str().unwrap(), None, None, None).unwrap();

    assert_ne!(first.summary.id, second.summary.id);
    assert!(packing::check_local(&index.resolve("model:example").unwrap()).is_err());
}

fn prepared(index: &ModelIndex, name: &str) -> (ModelEntry, String) {
    let model = index.resolve(name).unwrap();
    let pending = index.begin(ArtifactKind::Prepared).unwrap();

    std::fs::write(pending.lease.path.join("manifest.json"), "{}").unwrap();
    std::fs::write(pending.lease.path.join("experts.bin"), [1, 2, 3]).unwrap();

    let id = pending.artifact.id.clone();

    index
        .publish(&model, pending.artifact.clone(), None)
        .unwrap();

    (index.resolve(name).unwrap(), id)
}

#[test]
fn removal_and_gc_preserve_live_readers_and_external_sources() {
    let fixture = Fixture::registered("example");
    let index = &fixture.index;
    let source = &fixture.source;

    let (_, artifact) = prepared(index, "model:example");
    let lease = index.acquire_prepared("model:example").unwrap();

    index.remove("model:example", false).unwrap();
    assert_eq!(
        index.gc(false).unwrap().leased.as_slice(),
        std::slice::from_ref(&artifact)
    );
    assert!(lease.path.join("experts.bin").exists());
    drop(lease);

    let dry = index.gc(true).unwrap();

    assert_eq!(dry.artifacts.as_slice(), std::slice::from_ref(&artifact));
    assert!(index.artifact_root(&artifact).exists());
    index.gc(false).unwrap();
    assert!(!index.artifact_root(&artifact).exists());
    assert!(source.join("model.safetensors").exists());
}

#[test]
fn interrupted_publication_stays_unreferenced_and_can_be_collected() {
    let fixture = Fixture::registered("example");
    let index = &fixture.index;

    let model = index.resolve("model:example").unwrap();
    let pending = index.begin(ArtifactKind::Prepared).unwrap();

    index.remove("model:example", false).unwrap();
    assert!(
        index
            .publish(&model, pending.artifact.clone(), None)
            .is_err()
    );
    assert_eq!(index.gc(false).unwrap().leased.len(), 1);
    drop(pending);
    assert_eq!(index.gc(false).unwrap().artifacts.len(), 1);
}

#[test]
fn shared_artifacts_survive_removing_one_entry() {
    let fixture = Fixture::registered("example");
    let index = &fixture.index;

    let (mut model, artifact) = prepared(index, "model:example");
    model.id = id();
    model.name = Some("second".into());

    {
        let mut catalog = index.lock().unwrap();

        catalog.value.models.insert(model.id.clone(), model);
        catalog.save().unwrap();
    }

    index.remove("model:example", false).unwrap();
    assert!(index.gc(false).unwrap().artifacts.is_empty());
    assert!(index.artifact_root(&artifact).exists());
    assert_eq!(
        index.show("model:second").unwrap().artifacts[0].references,
        1
    );
}

#[test]
fn empty_listing_does_not_create_the_root_and_paths_remain_paths() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());

    assert!(index.list().unwrap().is_empty());
    assert!(!index.paths.data.exists());
    assert!(is_reference(Path::new("model:example")));
    assert!(!is_reference(Path::new("./model:example")));
    assert!(!is_reference(Path::new("/models/example")));
}

#[test]
fn incomplete_prepared_store_cannot_release_retained_source() {
    let fixture = Fixture::registered("example");
    let index = &fixture.index;

    prepared(index, "model:example");

    let model = index.resolve("model:example").unwrap();
    let packed = index.lock().unwrap().value.artifacts[model.prepared.as_ref().unwrap()].clone();
    let retained = index.begin(ArtifactKind::Source).unwrap();

    index
        .publish(&model, packed, Some(retained.artifact.clone()))
        .unwrap();
    assert!(index.remove("model:example", true).is_err());
    assert_eq!(
        index
            .resolve("model:example")
            .unwrap()
            .retained_source
            .as_deref(),
        Some(retained.artifact.id.as_str())
    );
}

#[test]
fn gc_does_not_follow_links_or_remove_external_artifacts() {
    let fixture = Fixture::new();
    let index = &fixture.index;
    let source = &fixture.source;
    let pending = index.begin(ArtifactKind::Prepared).unwrap();

    std::os::unix::fs::symlink(source, pending.lease.path.join("external")).unwrap();
    drop(pending);
    index.gc(false).unwrap();
    assert!(source.join("model.safetensors").exists());

    let artifact = Artifact {
        id: id(),
        kind: ArtifactKind::Prepared,
        external: Some(source.clone()),
        entry: Default::default(),
        ready: true,
    };

    {
        let mut catalog = index.lock().unwrap();

        catalog
            .value
            .artifacts
            .insert(artifact.id.clone(), artifact);
        catalog.save().unwrap();
    }

    assert_eq!(index.gc(false).unwrap().artifacts.len(), 1);
    assert!(source.join("model.safetensors").exists());
}

#[test]
fn invalid_catalog_paths_block_collection() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());
    let pending = index.begin(ArtifactKind::Source).unwrap();
    let file = pending.lease.path.join("keep");

    std::fs::write(&file, "data").unwrap();

    {
        let mut catalog = index.lock().unwrap();

        catalog
            .value
            .artifacts
            .get_mut(&pending.artifact.id)
            .unwrap()
            .entry = "../outside".into();

        catalog.save().unwrap();
    }

    drop(pending);
    assert!(index.gc(false).is_err());
    assert!(file.exists());
}

#[test]
fn registering_a_managed_prepared_path_reuses_its_ownership_record() {
    let fixture = Fixture::registered("example");
    let index = &fixture.index;
    let (model, artifact) = prepared(index, "model:example");
    let path = index.artifact_root(&artifact).canonicalize().unwrap();
    let source = Source::Local {
        path: path.clone(),
        fingerprint: local::fingerprint(&path).unwrap(),
    };

    index
        .register(source, model.description, Some("alias"), Some(path))
        .unwrap();

    assert_eq!(
        index.resolve("model:alias").unwrap().prepared.as_deref(),
        Some(artifact.as_str())
    );
    index.remove("model:example", false).unwrap();
    assert!(index.gc(false).unwrap().artifacts.is_empty());
    assert!(index.acquire_prepared("model:alias").is_ok());
}

#[test]
fn registering_a_managed_source_protects_it_during_import() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());
    let pending = index.begin(ArtifactKind::Source).unwrap();
    let source = checkpoint(&pending.lease.path);
    let path = source.canonicalize().unwrap();

    {
        let mut artifact = pending.artifact.clone();
        artifact.ready = true;
        artifact.entry = "source".into();
        let mut catalog = index.lock().unwrap();

        catalog
            .value
            .artifacts
            .insert(artifact.id.clone(), artifact);
        catalog.save().unwrap();
    }

    index
        .add(path.to_str().unwrap(), None, Some("alias"), None)
        .unwrap();
    drop(pending);
    assert!(index.gc(false).unwrap().artifacts.is_empty());

    let model = index.resolve("model:alias").unwrap();
    let lease = index
        .lease(&index.lock().unwrap().value.artifacts[model.retained_source.as_ref().unwrap()])
        .unwrap();

    index.remove("model:alias", false).unwrap();
    assert_eq!(index.gc(false).unwrap().leased.len(), 1);
    drop(lease);
    assert_eq!(index.gc(false).unwrap().artifacts.len(), 1);
    assert!(!path.exists());
}

mod reference;
mod regressions;

mod disk;
