use super::{Artifact, ArtifactKind, Catalog, ModelEntry, ModelIndex};
use anyhow::{Context, Result, ensure};
use std::{
    fs::File,
    path::{Path, PathBuf},
};

/// Keeps owned files reachable while a reader, mmap, or GPU operation uses them.
/// The caller must retain this handle until all derived views are finished.
pub struct ArtifactLease {
    /// Consumer directory, using the model root for legacy runtime layouts.
    pub path: PathBuf,
    _lock: Option<File>,
}

pub(super) struct Pending {
    pub artifact: Artifact,
    pub lease: ArtifactLease,
}

impl ModelIndex {
    pub(super) fn artifact_root(&self, id: &str) -> PathBuf {
        self.paths.data.join("artifacts").join(id)
    }

    pub(super) fn artifact_path(&self, artifact: &Artifact) -> PathBuf {
        artifact
            .external
            .clone()
            .unwrap_or_else(|| self.artifact_root(&artifact.id).join(&artifact.entry))
    }

    pub(super) fn lease_file(&self, id: &str) -> Result<File> {
        let dir = self.paths.data.join("leases");

        crate::storage::create_private_dir(&dir)?;

        Ok(File::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(dir.join(id))?)
    }

    pub(super) fn lease(&self, artifact: &Artifact) -> Result<ArtifactLease> {
        let lock = self.lease_file(&artifact.id)?;

        lock.lock_shared()?;

        let path = self.artifact_path(artifact);

        ensure!(
            path.exists(),
            "artifact {} is missing at {}",
            artifact.id,
            path.display()
        );

        Ok(ArtifactLease {
            path,
            _lock: Some(lock),
        })
    }

    /// Lease an existing, published prepared artifact without building or downloading.
    /// Fail if the model is unprepared or its artifact path is missing.
    pub fn acquire_prepared(&self, reference: &str) -> Result<ArtifactLease> {
        let catalog = self.lock()?;
        let model = super::lookup(&catalog.value, reference)?;
        let id = model.prepared.as_ref().with_context(|| {
            format!("model is not prepared; run `cherenkov prepare {reference}`")
        })?;
        let artifact = &catalog.value.artifacts[id];

        ensure!(artifact.ready, "artifact is not ready");

        self.lease(artifact)
    }

    pub(super) fn begin(&self, kind: ArtifactKind) -> Result<Pending> {
        let mut catalog = self.lock()?;
        let artifact = Artifact {
            id: super::id(),
            kind,
            external: None,
            entry: PathBuf::new(),
            ready: false,
        };
        let root = self.artifact_root(&artifact.id);
        let lock = self.lease_file(&artifact.id)?;

        lock.lock_shared()?;
        catalog
            .value
            .artifacts
            .insert(artifact.id.clone(), artifact.clone());
        catalog.save()?;
        crate::storage::create_private_dir(&root)?;

        Ok(Pending {
            artifact,
            lease: ArtifactLease {
                path: root,
                _lock: Some(lock),
            },
        })
    }

    pub(super) fn publish(
        &self,
        model: &ModelEntry,
        mut pending: Artifact,
        retained: Option<Artifact>,
    ) -> Result<()> {
        let mut catalog = self.lock()?;
        let current = catalog
            .value
            .models
            .get(&model.id)
            .context("model was removed during import")?;

        ensure!(
            current.prepared == model.prepared && current.retained_source == model.retained_source,
            "model changed during import; output was not published"
        );

        pending.ready = true;
        let prepared_id = pending.id.clone();

        catalog.value.artifacts.insert(prepared_id.clone(), pending);

        let retained_id = retained.map(|mut artifact| {
            artifact.ready = true;
            let id = artifact.id.clone();

            catalog.value.artifacts.insert(id.clone(), artifact);

            id
        });
        let entry = catalog.value.models.get_mut(&model.id).unwrap();
        entry.prepared = Some(prepared_id);

        if let Some(id) = retained_id {
            entry.retained_source = Some(id);
        }

        catalog.save()
    }
}

impl ArtifactLease {
    /// Select the model root for legacy `model/packed` stores with parent metadata.
    /// The original artifact lock remains held; only the runner's input path changes.
    pub(super) fn for_runtime(mut self) -> Self {
        self.path = self.runtime_path().to_owned();

        self
    }

    /// Locate legacy parent metadata without changing the artifact identity or lease.
    pub(super) fn runtime_path(&self) -> &Path {
        if self.path.join("config.json").is_file() && self.path.join("tokenizer.json").is_file() {
            return &self.path;
        }

        if self.path.file_name().is_none_or(|name| name != "packed") {
            return &self.path;
        }

        if let Some(parent) = self.path.parent()
            && parent.join("config.json").is_file()
            && parent.join("tokenizer.json").is_file()
        {
            return parent;
        }

        &self.path
    }

    /// Wrap a caller-owned path without checking it or acquiring a lock.
    /// The caller must keep its files unchanged while readers use them.
    pub fn external(path: PathBuf) -> Self {
        Self { path, _lock: None }
    }
}

pub(super) fn references(catalog: &Catalog, id: &str) -> usize {
    catalog
        .models
        .values()
        .filter(|m| m.prepared.as_deref() == Some(id) || m.retained_source.as_deref() == Some(id))
        .count()
}

/// Do not follow links: external targets cannot become GC-owned by traversal.
pub(super) fn bytes(path: &Path) -> Result<u64> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e.into()),
    };

    if !metadata.is_dir() {
        return Ok(metadata.len());
    }

    let mut total = 0_u64;

    for entry in std::fs::read_dir(path)? {
        total = total
            .checked_add(bytes(&entry?.path())?)
            .context("artifact size overflow")?;
    }

    Ok(total)
}

pub(super) fn precisions(path: &Path) -> Vec<u32> {
    let Ok(manifest) = crate::qwen4_exp::Manifest::load(path) else {
        return Vec::new();
    };
    let mut result = Vec::new();

    if path.join("manifest.json").is_file() && path.join("experts.bin").is_file() {
        result.push(4);
    }

    for bits in [3, 2] {
        if crate::qwen4_exp::lowbit::is_usable(path, &manifest.experts, bits).unwrap_or(false) {
            result.push(bits);
        }
    }

    result
}

pub(super) fn rebuild_variants(path: &Path, experts: &[u32], force: bool) -> Vec<u32> {
    let available = precisions(path);

    experts
        .iter()
        .copied()
        .filter(|&bits| bits != 4 && (force || !available.contains(&bits)))
        .collect()
}
