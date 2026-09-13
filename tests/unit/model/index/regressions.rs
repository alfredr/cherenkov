use super::*;
use crate::options::Options;
use std::os::unix::fs::{MetadataExt, symlink};

use crate::test_support::mlx_checkpoint as fixture;

struct Fixture {
    _dir: tempfile::TempDir,
    index: ModelIndex,
    source: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let index = index(dir.path());

        std::fs::create_dir(&source).unwrap();
        fixture::checkpoint(&source);
        index
            .add(source.to_str().unwrap(), None, Some("tiny"), None)
            .unwrap();

        Self {
            _dir: dir,
            index,
            source,
        }
    }

    fn pack(&self, experts: &[u32]) -> ArtifactLease {
        self.index
            .pack("model:tiny", request(experts, None))
            .unwrap();

        self.index.acquire_prepared("model:tiny").unwrap()
    }
}

fn request<'a>(experts: &'a [u32], output: Option<&'a Path>) -> PackOptions<'a> {
    PackOptions {
        output,
        experts,
        keep_source: false,
        token: None,
        repack: false,
    }
}

fn old_layout(path: &Path) {
    let file = path.join("manifest2.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    manifest["layout"] = "legacy".into();

    std::fs::write(file, serde_json::to_vec(&manifest).unwrap()).unwrap();
}

fn changed_codes(path: &Path) {
    let file = path.join("experts2.bin");
    let mut bytes = std::fs::read(&file).unwrap();
    bytes[0] ^= 0xff;

    std::fs::write(file, bytes).unwrap();
}

#[test]
fn automatic_rebuild_detaches_old_weights_but_reuses_unchanged_base() {
    let fixture = Fixture::new();
    let old = fixture.pack(&[2]);

    old_layout(&old.path);
    changed_codes(&old.path);

    let before = std::fs::read(old.path.join("experts2.bin")).unwrap();
    let current = fixture.pack(&[2, 3]);

    assert_eq!(
        std::fs::read(old.path.join("experts2.bin")).unwrap(),
        before
    );
    assert!(!old.path.join("experts3.bin").exists());
    assert_ne!(
        inode(&old.path, "experts2.bin"),
        inode(&current.path, "experts2.bin")
    );
    assert_eq!(
        inode(&old.path, "experts.bin"),
        inode(&current.path, "experts.bin")
    );
    assert_eq!(store::precisions(&current.path), [4, 3, 2]);
}

fn inode(path: &Path, name: &str) -> u64 {
    std::fs::metadata(path.join(name)).unwrap().ino()
}

#[test]
fn packing_repairs_invalid_variants_instead_of_accepting_filenames() {
    let mutations: [fn(&Path); 3] = [old_layout, changed_codes, |path| {
        std::fs::write(path.join("experts2.bin"), [0]).unwrap();
    }];

    for mutate in mutations {
        let fixture = Fixture::new();
        let old = fixture.pack(&[2]);

        mutate(&old.path);
        assert_eq!(
            fixture.index.show("model:tiny").unwrap().summary.precisions,
            [4]
        );

        let current = fixture.pack(&[2]);

        assert_ne!(current.path, old.path);
        assert_eq!(store::precisions(&current.path), [4, 2]);
    }
}

#[test]
fn runtime_repairs_external_variants_only_when_policy_allows() {
    let fixture = Fixture::new();

    crate::qwen4_exp::pack::prepare(&fixture.source, None, &[2]).unwrap();

    let external = fixture.source.join("packed");

    old_layout(&external);
    fixture
        .index
        .add(external.to_str().unwrap(), None, Some("external"), None)
        .unwrap();

    let mut options = Options {
        experts: 2,
        build_missing_store: false,
        ..Options::default()
    };
    let reference = Path::new("model:external");

    assert!(resolve_runtime(fixture.index.paths.clone(), reference, &mut options).is_err());
    assert!(
        fixture
            .index
            .list()
            .unwrap()
            .iter()
            .all(|model| model.owned_bytes == 0)
    );

    options.build_missing_store = true;

    let resolved = resolve_runtime(fixture.index.paths.clone(), reference, &mut options).unwrap();

    assert_ne!(resolved.path, external);
    assert_eq!(store::precisions(&resolved.path), [4, 2]);
    assert_eq!(store::precisions(&external), [4]);
    assert!(!options.build_missing_store);
}

#[test]
fn reregistering_adopts_newly_prepared_output_without_changing_id() {
    let fixture = Fixture::new();
    let before = fixture.index.resolve("model:tiny").unwrap();

    crate::qwen4_exp::pack::prepare(&fixture.source, None, &[4]).unwrap();

    let after = fixture
        .index
        .add(fixture.source.to_str().unwrap(), None, Some("tiny"), None)
        .unwrap();
    let lease = fixture.index.acquire_prepared("model:tiny").unwrap();

    assert_eq!(after.summary.id, before.id);
    assert!(after.summary.prepared);
    assert_eq!(
        lease.path,
        fixture.source.join("packed").canonicalize().unwrap()
    );

    let managed = fixture.pack(&[2]);

    fixture
        .index
        .add(fixture.source.to_str().unwrap(), None, None, None)
        .unwrap();
    assert_eq!(
        fixture.index.acquire_prepared("model:tiny").unwrap().path,
        managed.path
    );
}

#[test]
fn nested_exports_are_rejected_even_through_a_symlink() {
    let fixture = Fixture::new();
    let old = fixture.pack(&[4]);
    let alias = fixture.source.join("alias");

    symlink(&old.path, &alias).unwrap();

    for parent in [&old.path, &alias] {
        let output = parent.join("export");
        let result = fixture
            .index
            .pack("model:tiny", request(&[4], Some(&output)));

        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("inside managed artifacts")
        );
        assert!(!output.exists());
    }

    fixture.index.gc(false).unwrap();
    assert_eq!(
        fixture.index.acquire_prepared("model:tiny").unwrap().path,
        old.path
    );
}

#[test]
fn alternate_paths_inside_a_managed_source_cannot_be_registered_as_external() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path());
    let mut pending = index.begin(ArtifactKind::Source).unwrap();
    let snapshot = checkpoint(&pending.lease.path);
    let entry = pending.lease.path.join("entry");

    std::fs::create_dir(&entry).unwrap();

    pending.artifact.entry = "entry".into();
    pending.artifact.ready = true;

    {
        let mut catalog = index.lock().unwrap();

        catalog
            .value
            .artifacts
            .insert(pending.artifact.id.clone(), pending.artifact.clone());
        catalog.save().unwrap();
    }

    assert!(
        index
            .add(snapshot.to_str().unwrap(), None, Some("alias"), None)
            .is_err()
    );
    assert!(index.list().unwrap().is_empty());
}

#[test]
fn concurrent_exports_claim_one_destination_without_mixing_stores() {
    let fixture = Fixture::new();
    let second = fixture.source.join("second");
    let output = fixture.source.join("export");

    std::fs::create_dir(&second).unwrap();
    fixture::checkpoint(&second);
    fixture
        .index
        .add(second.to_str().unwrap(), None, Some("second"), None)
        .unwrap();

    let barrier = std::sync::Barrier::new(2);
    let successes = std::thread::scope(|scope| {
        let workers: Vec<_> = ["model:tiny", "model:second"]
            .into_iter()
            .map(|reference| {
                let index = &fixture.index;
                let output = &output;
                let barrier = &barrier;

                scope.spawn(move || {
                    barrier.wait();

                    index.pack(reference, request(&[4], Some(output))).is_ok()
                })
            })
            .collect();

        workers
            .into_iter()
            .map(|worker| usize::from(worker.join().unwrap()))
            .sum::<usize>()
    });

    assert_eq!(successes, 1);
    assert!(Checkpoint::open(&output).is_ok());
}

#[test]
fn metadata_repair_publishes_without_mutating_leased_weights_or_existing_templates() {
    let fixture = Fixture::new();

    std::fs::write(
        fixture.source.join("chat_template.jinja"),
        "source template",
    )
    .unwrap();
    std::fs::write(fixture.source.join("tokenizer_config.json"), "{}").unwrap();
    // Register this source version before building the artifact that will need repair.
    fixture.index.remove("tiny", false).unwrap();
    fixture
        .index
        .add(fixture.source.to_str().unwrap(), None, Some("tiny"), None)
        .unwrap();

    let old = fixture.pack(&[4]);

    std::fs::remove_file(old.path.join("tokenizer_config.json")).unwrap();
    std::fs::write(old.path.join("chat_template.jinja"), "retained template").unwrap();

    let current = fixture.pack(&[4]);

    assert_ne!(old.path, current.path);
    assert!(!old.path.join("tokenizer_config.json").exists());
    assert_eq!(
        std::fs::read_to_string(current.path.join("tokenizer_config.json")).unwrap(),
        "{}"
    );
    assert_eq!(
        std::fs::read_to_string(current.path.join("chat_template.jinja")).unwrap(),
        "retained template"
    );
    assert_eq!(
        inode(&old.path, "experts.bin"),
        inode(&current.path, "experts.bin")
    );
    assert_eq!(fixture.pack(&[4]).path, current.path);
}

#[test]
fn legacy_packed_registration_preserves_parent_notices_and_existing_metadata() {
    let fixture = Fixture::new();

    crate::qwen4_exp::pack::prepare(&fixture.source, None, &[4]).unwrap();

    let packed = fixture.source.join("packed");
    let notices = [
        "LICENSE",
        "LICENSE.txt",
        "LICENSE.md",
        "NOTICE",
        "NOTICE.txt",
        "NOTICE.md",
    ];

    for name in notices {
        std::fs::write(fixture.source.join(name), name).unwrap();
    }

    std::fs::write(
        fixture.source.join("chat_template.jinja"),
        "parent template",
    )
    .unwrap();
    std::fs::write(packed.join("chat_template.jinja"), "prepared template").unwrap();
    std::fs::remove_file(packed.join("tokenizer.json")).unwrap();

    // Keep a distinct, valid prepared configuration while tokenizer lookup uses the parent.
    let mut config = std::fs::read(packed.join("config.json")).unwrap();

    config.push(b'\n');
    std::fs::write(packed.join("config.json"), &config).unwrap();
    fixture
        .index
        .add(packed.to_str().unwrap(), None, Some("legacy"), None)
        .unwrap();

    let old = fixture.index.acquire_prepared("legacy").unwrap();

    fixture
        .index
        .pack("legacy", request(&[4, 2], None))
        .unwrap();

    let current = fixture.index.acquire_prepared("legacy").unwrap();

    assert_ne!(old.path, current.path);
    assert_eq!(store::precisions(&current.path), [4, 2]);
    assert_eq!(
        std::fs::read(current.path.join("config.json")).unwrap(),
        config
    );
    assert_eq!(
        std::fs::read_to_string(current.path.join("chat_template.jinja")).unwrap(),
        "prepared template"
    );
    assert_eq!(
        std::fs::read(current.path.join("tokenizer.json")).unwrap(),
        std::fs::read(fixture.source.join("tokenizer.json")).unwrap()
    );

    for name in notices {
        assert_eq!(
            std::fs::read_to_string(current.path.join(name)).unwrap(),
            name
        );
        assert!(!old.path.join(name).exists());
    }

    assert!(!old.path.join("tokenizer.json").exists());
    assert_eq!(std::fs::read(old.path.join("config.json")).unwrap(), config);
}
