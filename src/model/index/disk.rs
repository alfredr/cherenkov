//! Registered filesystem stores preserve physical layouts and external ownership.

use super::{Catalog, ModelIndex};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Physical layout used to find an owner/repository within a disk store.
#[derive(Clone, Copy, Default, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum DiskLayout {
    /// Directories at ROOT/owner/repo, as used by ordinary and LM Studio stores.
    #[default]
    Directory,
    /// Hugging Face's models--owner--repo/snapshots/commit cache layout.
    HfCache,
}

/// A named, externally owned filesystem store registered in this index.
#[derive(Clone, Serialize, Deserialize)]
pub struct DiskStore {
    /// Stable identity; removing and re-registering a name creates a new identity.
    pub id: String,
    /// URI authority used in disk://NAME/owner/repo.
    pub name: String,
    /// Canonical filesystem root. Registration never takes ownership of its files.
    pub path: PathBuf,
    /// Model-location convention beneath the root.
    pub layout: DiskLayout,
    /// Whether source URI resolution is enabled; existing index entries are retained.
    pub enabled: bool,
}

impl ModelIndex {
    /// Register an existing filesystem root without crawling or importing models.
    pub fn add_disk_store(&self, name: &str, path: &Path, layout: DiskLayout) -> Result<DiskStore> {
        ensure!(
            crate::storage::safe_component(name),
            "invalid disk store name"
        );

        let path = path.canonicalize().context("opening disk store")?;

        ensure!(path.is_dir(), "disk store root must be a directory");

        let mut catalog = self.lock()?;

        ensure!(
            !catalog.value.stores.contains_key(name),
            "store {name:?} is already registered"
        );

        let store = DiskStore {
            id: super::id(),
            name: name.to_owned(),
            path,
            layout,
            enabled: true,
        };

        catalog.value.stores.insert(name.to_owned(), store.clone());
        catalog.save()?;

        Ok(store)
    }

    /// List store registrations without opening their roots or contacting providers.
    pub fn disk_stores(&self) -> Result<Vec<DiskStore>> {
        if !self.paths.data.join("index.json").exists() {
            return Ok(Vec::new());
        }

        Ok(self.lock()?.value.stores.values().cloned().collect())
    }

    /// Enable or disable resolution through a store's URI without changing its ID.
    pub fn set_disk_store_enabled(&self, name: &str, enabled: bool) -> Result<()> {
        let mut catalog = self.lock()?;
        catalog
            .value
            .stores
            .get_mut(name)
            .context("store is not registered")?
            .enabled = enabled;

        catalog.save()
    }

    /// Forget a store registration without deleting files or registered model entries.
    pub fn remove_disk_store(&self, name: &str) -> Result<()> {
        let mut catalog = self.lock()?;

        ensure!(
            catalog.value.stores.remove(name).is_some(),
            "store is not registered"
        );

        catalog.save()
    }
}

/// Find an enabled store before resolving a source or inspecting a registered URI.
pub(super) fn registered<'a>(catalog: &'a Catalog, name: &str) -> Result<&'a DiskStore> {
    let store = catalog
        .stores
        .get(name)
        .context("disk store is not registered")?;

    ensure!(store.enabled, "disk store {name:?} is disabled");

    Ok(store)
}

impl DiskStore {
    /// Resolve a source inside this root, rejecting directory traversal and symlink escapes.
    pub(super) fn locate(&self, repo: &str, revision: Option<&str>) -> Result<PathBuf> {
        let candidate = match self.layout {
            DiskLayout::Directory => self.path.join(repo),
            DiskLayout::HfCache => self.snapshot(repo, revision)?,
        };
        let path = candidate
            .canonicalize()
            .context("model is absent from the disk store")?;

        ensure!(
            path.starts_with(&self.path),
            "model path escapes the disk store"
        );

        Ok(path)
    }

    /// Select a cached revision without downloading or advancing a cached ref.
    fn snapshot(&self, repo: &str, revision: Option<&str>) -> Result<PathBuf> {
        let root = self
            .path
            .join(format!("models--{}", repo.replace('/', "--")));
        let snapshots = root.join("snapshots");
        let requested = revision.unwrap_or("main");

        ensure!(
            !Path::new(requested).is_absolute()
                && requested.split('/').all(crate::storage::safe_component),
            "invalid cache revision"
        );

        let cached_ref = match std::fs::read_to_string(root.join("refs").join(requested)) {
            Ok(commit) => Some(commit),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error).context("reading cached revision"),
        };

        if let Some(commit) = cached_ref {
            let commit = commit.trim();

            ensure!(
                super::reference::is_revision(commit),
                "invalid cached commit"
            );

            return Ok(snapshots.join(commit));
        }

        ensure!(
            revision.is_none_or(super::reference::is_revision),
            "cached ref not found; commit prefixes require at least eight hexadecimal characters"
        );

        let mut matches = std::fs::read_dir(&snapshots)?
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|entry| {
                entry.file_name().to_str().is_some_and(|name| {
                    super::reference::is_revision(name)
                        && revision.is_none_or(|rev| name.starts_with(rev))
                })
            });
        let first = matches.next().context("cached revision not found")?;

        ensure!(
            matches.next().is_none(),
            "cached revision is ambiguous; append @commit"
        );

        Ok(first.path())
    }

    /// Map a known local path to the same selector type produced by URL parsing.
    pub(super) fn selector(&self, path: &Path) -> Option<super::selector::Selector> {
        let relative = path.strip_prefix(&self.path).ok()?;
        let parts: Vec<_> = relative
            .iter()
            .map(|part| part.to_str())
            .collect::<Option<_>>()?;
        let (repo, revision) = match (self.layout, parts.as_slice()) {
            (DiskLayout::Directory, [owner, repo]) => (format!("{owner}/{repo}"), None),
            (DiskLayout::HfCache, [repo, "snapshots", commit]) => {
                let parts: Vec<_> = repo.split("--").collect();
                let ["models", owner, repo] = parts.as_slice() else {
                    return None;
                };

                (format!("{owner}/{repo}"), Some((*commit).to_owned()))
            }
            _ => return None,
        };

        Some(super::selector::Selector::Disk {
            store: self.name.clone(),
            repo,
            revision,
        })
    }
}
