//! Persistent model references and explicit ownership of prepared/source stores.
mod catalog;
mod disk;
mod gc;
mod hub;
mod local;
mod packing;
mod records;
mod reference;
mod resolve;
mod selector;
pub use disk::{DiskLayout, DiskStore};
pub use resolve::ResolveOptions;
mod store;

use crate::{
    model::{Checkpoint, ModelDescription},
    storage::Paths,
};
use anyhow::{Context, Result, ensure};
pub use packing::PackOptions;
use records::{Artifact, Catalog};
pub use records::{
    ArtifactDetails, ArtifactKind, GcReport, ModelDetails, ModelEntry, ModelSummary, Removal,
    Source,
};
use reference::lookup;
use std::path::Path;
pub use store::ArtifactLease;

/// Persistent model references and artifact ownership under one storage root.
/// Operations lock the catalog as needed; this handle does not retain artifact leases.
#[derive(Clone)]
pub struct ModelIndex {
    pub(super) paths: Paths,
}

impl ModelIndex {
    /// Select the index location without opening or creating it.
    pub fn new(paths: Paths) -> Self {
        Self { paths }
    }

    /// Look up a source reference, alias, or model ID and return its catalog entry.
    /// The returned metadata does not keep its artifacts alive.
    pub fn resolve(&self, reference: &str) -> Result<ModelEntry> {
        Ok(lookup(&self.lock()?.value, reference)?.clone())
    }

    /// Register a local checkpoint path or an `hf://owner/repo` source.
    /// Hub registration pins a commit and reads metadata without downloading the
    /// full weights. `revision` defaults to `main`; it and `token` apply only to HF.
    /// Registration does not transfer file ownership. The same source reuses its ID.
    pub fn add(
        &self,
        source: &str,
        revision: Option<&str>,
        name: Option<&str>,
        token: Option<&str>,
    ) -> Result<ModelDetails> {
        self.select(
            Path::new(source),
            ResolveOptions {
                name,
                revision,
                token,
            },
        )
    }

    /// Inspect a local source and adopt existing prepared output without copying it.
    fn add_local(&self, path: &Path, name: Option<&str>) -> Result<ModelDetails> {
        let path = std::fs::canonicalize(path).context("opening local checkpoint")?;
        let description = Checkpoint::open(&path)?.description;
        let fingerprint = local::fingerprint(&path)?;
        let prepared = if path.join("manifest.json").is_file() {
            Some(path.clone())
        } else {
            path.join("packed/manifest.json")
                .is_file()
                .then(|| path.join("packed"))
        };
        let source = Source::Local { path, fingerprint };
        let id = self.register(source, description, name, prepared)?;

        self.show(&id)
    }

    fn register(
        &self,
        source: Source,
        description: ModelDescription,
        name: Option<&str>,
        prepared: Option<std::path::PathBuf>,
    ) -> Result<String> {
        let mut catalog = self.lock()?;
        let same = catalog
            .value
            .models
            .values()
            .find(|m| same_source(&m.source, &source))
            .map(|m| m.id.clone());

        if let Some(name) = name {
            let owner = catalog
                .value
                .models
                .values()
                .find(|m| m.name.as_deref() == Some(name));

            ensure!(
                owner.is_none_or(|m| Some(&m.id) == same.as_ref()),
                "model name {name:?} is already registered"
            );
        }

        let mut model = match same {
            Some(id) => catalog.value.models[&id].clone(),
            None => ModelEntry {
                id: id(),
                name: None,
                source,
                description,
                prepared: None,
                retained_source: None,
            },
        };

        if let Some(name) = name {
            ensure!(
                model.name.as_deref().is_none_or(|current| current == name),
                "source already has name {:?}",
                model.name
            );

            model.name = Some(name.to_owned());
        }

        if let Source::Local { path, .. } = &model.source {
            self.require_registered_location(&catalog.value, path)?;

            model.retained_source = model
                .retained_source
                .or_else(|| self.artifact_at(&catalog.value, ArtifactKind::Source, path));
        }

        if model.prepared.is_none()
            && let Some(path) = prepared
        {
            model.prepared = Some(self.adopt_prepared(&mut catalog.value, &path)?);
        }

        let id = model.id.clone();

        catalog.value.models.insert(id.clone(), model);
        catalog.save()?;

        Ok(id)
    }

    fn adopt_prepared(&self, catalog: &mut Catalog, path: &Path) -> Result<String> {
        let path = path.canonicalize()?;

        if let Some(id) = self.artifact_at(catalog, ArtifactKind::Prepared, &path) {
            return Ok(id);
        }

        self.require_external_path(&path)?;

        let artifact = Artifact {
            id: id(),
            kind: ArtifactKind::Prepared,
            external: Some(path),
            entry: Default::default(),
            ready: true,
        };
        let id = artifact.id.clone();

        catalog.artifacts.insert(id.clone(), artifact);

        Ok(id)
    }

    /// An exact managed entry reuses its ownership record. Other paths inside
    /// the artifact namespace cannot safely be recorded as external locations.
    fn require_registered_location(&self, catalog: &Catalog, path: &Path) -> Result<()> {
        if self
            .artifact_at(catalog, ArtifactKind::Source, path)
            .is_some()
            || self
                .artifact_at(catalog, ArtifactKind::Prepared, path)
                .is_some()
        {
            return Ok(());
        }

        self.require_external_path(path)
    }

    fn require_external_path(&self, canonical_path: &Path) -> Result<()> {
        let artifacts = self.paths.data.join("artifacts");
        let root = match artifacts.canonicalize() {
            Ok(root) => root,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };

        ensure!(
            !canonical_path.starts_with(root),
            "path is inside managed artifacts; use an indexed model reference or a location outside the artifacts directory"
        );

        Ok(())
    }

    /// Registering a managed path adds a reference to its existing ownership record.
    fn artifact_at(&self, catalog: &Catalog, kind: ArtifactKind, path: &Path) -> Option<String> {
        catalog
            .artifacts
            .values()
            .find(|artifact| {
                artifact.ready
                    && artifact.kind == kind
                    && self.artifact_path(artifact).canonicalize().ok().as_deref() == Some(path)
            })
            .map(|artifact| artifact.id.clone())
    }

    /// Summarize registered models and inspect their current artifact availability.
    /// An absent index returns an empty list without creating it.
    pub fn list(&self) -> Result<Vec<ModelSummary>> {
        // An empty index can be listed on a read-only root without creating it.
        if !self.paths.data.join("index.json").exists() {
            return Ok(Vec::new());
        }

        let catalog = self.lock()?;

        catalog
            .value
            .models
            .values()
            .map(|m| self.details(&catalog.value, m).map(|d| d.summary))
            .collect()
    }

    /// Report a model's source, preparation requirements, and referenced artifacts.
    pub fn show(&self, reference: &str) -> Result<ModelDetails> {
        let catalog = self.lock()?;

        self.details(&catalog.value, lookup(&catalog.value, reference)?)
    }

    /// Inspect the referenced prepared artifact, or return registered source metadata.
    /// A recorded prepared artifact that is missing or unreadable returns an error.
    pub fn description(&self, reference: &str) -> Result<ModelDescription> {
        let entry = self.resolve(reference)?;

        if entry.prepared.is_some() {
            let lease = self.acquire_prepared(reference)?;

            return Ok(Checkpoint::open(&lease.path)?.description);
        }

        Ok(entry.description)
    }

    fn details(&self, catalog: &Catalog, model: &ModelEntry) -> Result<ModelDetails> {
        let mut artifacts = Vec::new();
        let mut owned_bytes = 0;
        let mut external_bytes = 0;
        let mut precisions = Vec::new();

        for id in model.prepared.iter().chain(&model.retained_source) {
            let artifact = &catalog.artifacts[id];
            let path = self.artifact_path(artifact);
            let owned = artifact.external.is_none();
            let size_path = if owned {
                self.artifact_root(id)
            } else {
                path.clone()
            };
            let bytes = store::bytes(&size_path)?;

            if owned {
                owned_bytes += bytes;
            } else {
                external_bytes += bytes;
            }

            if artifact.kind == ArtifactKind::Prepared {
                precisions = store::precisions(&path);
            }

            artifacts.push(ArtifactDetails {
                id: id.clone(),
                kind: artifact.kind,
                available: path.exists(),
                path,
                owned,
                bytes,
                references: store::references(catalog, id),
            });
        }

        let source_local = match &model.source {
            Source::Local { path, .. } => path.exists(),
            Source::HuggingFace { .. } => artifacts
                .iter()
                .any(|a| a.kind == ArtifactKind::Source && a.available),
        };
        let prepared = artifacts
            .iter()
            .any(|a| a.kind == ArtifactKind::Prepared && a.available);

        Ok(ModelDetails {
            summary: ModelSummary {
                id: model.id.clone(),
                reference: reference::preferred(catalog, model),
                name: model.name.clone(),
                architecture: model.description.architecture.clone(),
                source: model.source.clone(),
                source_local,
                prepared,
                precisions,
                owned_bytes,
                external_bytes,
            },
            preparation: if prepared {
                crate::model::Preparation::Direct
            } else {
                model.description.preparation()
            },
            artifacts,
        })
    }
}

/// Parse an alias, ID, or source URI without resolving it. Explicit paths return false.
pub fn is_reference(value: &Path) -> bool {
    selector::parse(value).is_ok_and(|selector| !matches!(selector, selector::Selector::Path(_)))
}

/// Anchor explicit relative paths while preserving parsed aliases and source URIs.
/// Invalid locators return an error instead of becoming filesystem paths.
pub fn anchor_selector(value: &Path, base: &Path) -> Result<std::path::PathBuf> {
    Ok(match selector::parse(value)? {
        selector::Selector::Path(path) => base.join(path),
        _ => value.to_owned(),
    })
}

/// Validate selector syntax without resolving an entry, reading files, or registering.
pub fn validate_selector(value: &str) -> Result<()> {
    selector::parse(Path::new(value)).map(|_| ())
}

/// List available expert precisions in descending order after loading the manifest.
/// Q4 requires an existing base file. Q2/Q3 also check layout, size, and sample
/// records. A manifest load failure returns an empty list.
pub fn prepared_precisions(path: &Path) -> Vec<u32> {
    store::precisions(path)
}

/// Resolve through the index and lease the prepared artifact.
/// Explicit source paths are registered if needed; files remain externally owned.
pub fn resolve_path(paths: Paths, path: &Path) -> Result<ArtifactLease> {
    let index = ModelIndex::new(paths);
    let selected = index.select(path, ResolveOptions::default())?;

    index.acquire_prepared(&selected.summary.id)
}

/// Prepare missing variants through publication, never by mutating a leased view.
/// Indexed models must already have a prepared base. If the options permit it,
/// missing variants are built before returning a lease to the published store.
/// On success, indexed loads disable later repacking and store creation in `options`.
/// Legacy `model/packed` stores use their model root to find config and tokenizer.
pub fn resolve_runtime(
    paths: Paths,
    path: &Path,
    options: &mut crate::options::Options,
) -> Result<ArtifactLease> {
    let index = ModelIndex::new(paths);
    let selected = index.select(path, ResolveOptions::default())?;
    let reference = selected.summary.id.as_str();
    let lease = index.acquire_prepared(reference)?;
    let experts = [
        options.experts,
        options.miss_experts.unwrap_or(options.experts),
    ];
    let available = prepared_precisions(&lease.path);
    let missing = experts.iter().any(|bits| !available.contains(bits));

    if missing || options.repack {
        ensure!(
            options.build_missing_store,
            "selected expert stores are missing; run `cherenkov prepare {reference} --experts {},{}`",
            experts[0],
            experts[1]
        );
        index.pack(
            reference,
            PackOptions {
                output: None,
                experts: &experts,
                keep_source: false,
                token: None,
                repack: options.repack,
            },
        )?;
    }

    options.repack = false;
    options.build_missing_store = false;

    Ok(index.acquire_prepared(reference)?.for_runtime())
}

fn id() -> String {
    format!("{:032x}", rand::random::<u128>())
}

fn validate_name(name: Option<&str>) -> Result<()> {
    if let Some(name) = name {
        ensure!(
            crate::storage::safe_component(name) && name.len() <= 64,
            "model names must contain 1-64 letters, digits, '.', '_' or '-'"
        );
        ensure!(
            !(name.len() == 32 && name.bytes().all(|b| b.is_ascii_hexdigit())),
            "model names cannot look like an ID"
        );
    }

    Ok(())
}

fn same_source(a: &Source, b: &Source) -> bool {
    match (a, b) {
        (
            Source::Local {
                path: a,
                fingerprint: x,
            },
            Source::Local {
                path: b,
                fingerprint: y,
            },
        ) => a == b && x == y,
        (
            Source::HuggingFace {
                repo: a,
                revision: x,
                endpoint: p,
            },
            Source::HuggingFace {
                repo: b,
                revision: y,
                endpoint: q,
            },
        ) => a == b && x == y && p == q,
        _ => false,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/model/index/mod.rs"]
mod tests;
