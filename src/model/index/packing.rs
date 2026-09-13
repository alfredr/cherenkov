use super::{
    Artifact, ArtifactKind, ArtifactLease, ModelDetails, ModelEntry, ModelIndex, Source, local,
    store::Pending,
};
use crate::{
    download,
    model::{Checkpoint, Preparation},
    qwen4_exp,
    storage::Paths,
};
use anyhow::{Context, Result, ensure};
use std::{
    os::unix::fs::DirBuilderExt,
    path::{Path, PathBuf},
};

/// Requested expert variants and source retention for an indexed model.
pub struct PackOptions<'a> {
    /// Export to a caller-owned directory instead of selecting the managed store.
    /// A changed or new export must use a new directory outside managed artifacts.
    pub output: Option<&'a Path>,
    /// Nonempty set of requested expert bit widths: 2, 3, or 4.
    pub experts: &'a [u32],
    /// Retain a downloaded HF source after successful preparation.
    pub keep_source: bool,
    /// Optional HF credential for source downloads; never written to the catalog.
    pub token: Option<&'a str>,
    /// Force rebuilding the requested low-bit variants even if they are usable.
    pub repack: bool,
}

impl ModelIndex {
    /// Prepare requested variants and atomically update the model's artifact reference.
    /// Existing usable stores can be reused. Rebuilds leave previously leased stores
    /// unchanged. HF imports use a full local snapshot; newly downloaded sources are
    /// discarded after success unless retention was requested.
    pub fn pack(&self, reference: &str, request: PackOptions<'_>) -> Result<ModelDetails> {
        let PackOptions {
            output,
            experts,
            keep_source,
            token,
            repack,
        } = request;

        ensure!(
            !experts.is_empty() && experts.iter().all(|b| (2..=4).contains(b)),
            "expert precisions must be 4,3,2"
        );

        let model = self.resolve(reference)?;

        ensure!(
            !matches!(
                model.description.preparation(),
                Preparation::Unsupported { .. }
            ),
            "model cannot be prepared by this engine"
        );

        let previous = self.prepared_copy(&model)?;
        let retain_download = keep_source
            && matches!(&model.source, Source::HuggingFace { .. })
            && model.retained_source.is_none();

        let metadata_source =
            self.metadata_repair(&model, previous.as_ref().map(|(_, lease)| lease))?;

        if let Some((artifact, lease)) = &previous {
            let bits = super::store::precisions(&lease.path);
            let same_output =
                output.is_none_or(|p| p.canonicalize().ok() == lease.path.canonicalize().ok());

            if !repack
                && experts.iter().all(|b| bits.contains(b))
                && same_output
                && !retain_download
                && metadata_source.is_none()
            {
                return self.show(reference);
            }

            ensure!(
                output.is_none_or(|p| !p.exists()),
                "--output must name a new directory when adding variants"
            );
            ensure!(artifact.ready, "previous artifact is incomplete");
        }

        let pending = self.begin(ArtifactKind::Prepared)?;
        let mut downloaded = None;
        let mut source_lease = None;

        if let Some((artifact, lease)) = &previous {
            if artifact.external.is_some() {
                crate::storage::require_space(
                    &pending.lease.path,
                    super::store::bytes(&lease.path)?,
                )?;
            }

            local::copy_store(
                &lease.path,
                &pending.lease.path,
                artifact.external.is_none(),
            )?;

            copy_runtime_metadata(lease, &pending.lease.path)?;

            if let Some(source) = &metadata_source {
                qwen4_exp::pack::copy_chat_metadata(&source.path, &pending.lease.path)?;
            }

            // Both explicit and automatic rebuilds must detach shared inodes.
            let rebuild = super::store::rebuild_variants(&pending.lease.path, experts, repack);

            remove_variants(&pending.lease.path, &rebuild)?;

            qwen4_exp::pack::prepare(&pending.lease.path, None, experts)?;

            if retain_download {
                source_lease = Some(self.acquire_source(&model, token, &mut downloaded)?);
            }
        } else {
            let input = self.acquire_source(&model, token, &mut downloaded)?;

            qwen4_exp::pack::prepare(&input.path, Some(&pending.lease.path), experts)?;
            check_local(&model)?;

            source_lease = Some(input);
        }

        Checkpoint::open(&pending.lease.path).context("validating prepared output")?;
        sync_files(&pending.lease.path)?;

        let artifact = match output {
            Some(path) => self.export(&pending, path)?,
            None => pending.artifact.clone(),
        };
        let retained = keep_source
            .then(|| downloaded.as_ref().map(|p: &Pending| p.artifact.clone()))
            .flatten();

        self.publish(&model, artifact, retained)?;
        drop(source_lease);

        if let Some(source) = downloaded {
            let id = source.artifact.id.clone();

            drop(source);

            if !keep_source {
                self.discard(&id)?;
            }
        }

        let id = pending.artifact.id.clone();

        drop(pending);

        if output.is_some() {
            self.discard(&id)?;
        }

        self.show(reference)
    }

    fn prepared_copy(&self, model: &ModelEntry) -> Result<Option<(Artifact, ArtifactLease)>> {
        let Some(id) = &model.prepared else {
            return Ok(None);
        };
        let catalog = self.lock()?;
        let artifact = catalog
            .value
            .artifacts
            .get(id)
            .context("prepared artifact disappeared")?;

        if !self.artifact_path(artifact).exists() {
            return Ok(None);
        }

        Ok(Some((artifact.clone(), self.lease(artifact)?)))
    }

    /// Hold the source only when a prepared artifact needs missing metadata copied.
    fn metadata_repair(
        &self,
        model: &ModelEntry,
        previous: Option<&ArtifactLease>,
    ) -> Result<Option<ArtifactLease>> {
        let Some(previous) = previous else {
            return Ok(None);
        };
        let Some(source) = self.available_metadata(model)? else {
            return Ok(None);
        };

        if !qwen4_exp::pack::missing_chat_metadata(&source.path, &previous.path) {
            return Ok(None);
        }

        check_local(model)?;

        Ok(Some(source))
    }

    /// Retain available source metadata without downloading another weight snapshot.
    fn available_metadata(&self, model: &ModelEntry) -> Result<Option<ArtifactLease>> {
        if let Source::Local { path, .. } = &model.source {
            return path
                .is_dir()
                .then(|| self.local_source_lease(model, path))
                .transpose();
        }

        let Some(id) = &model.retained_source else {
            return Ok(None);
        };
        let catalog = self.lock()?;
        let artifact = catalog
            .value
            .artifacts
            .get(id)
            .context("source artifact disappeared")?;

        if !self.artifact_path(artifact).is_dir() {
            return Ok(None);
        }

        Ok(Some(self.lease(artifact)?))
    }

    fn acquire_source(
        &self,
        model: &ModelEntry,
        token: Option<&str>,
        downloaded: &mut Option<Pending>,
    ) -> Result<ArtifactLease> {
        if let Source::Local { path, .. } = &model.source {
            let lease = self.local_source_lease(model, path)?;

            check_local(model)?;

            return Ok(lease);
        }

        if let Some(id) = &model.retained_source {
            let catalog = self.lock()?;

            if let Some(artifact) = catalog.value.artifacts.get(id)
                && self.artifact_path(artifact).exists()
            {
                return self.lease(artifact);
            }
        }

        let Source::HuggingFace {
            repo,
            revision,
            endpoint,
        } = &model.source
        else {
            unreachable!()
        };
        let mut pending = self.begin(ArtifactKind::Source)?;
        let root = pending.lease.path.clone();
        let paths = Paths {
            data: root.clone(),
            scratch: root.join("transfer"),
            config: root.join("unused.toml"),
        };
        let path = download::run_at(
            &paths,
            download::Download {
                repo,
                revision,
                token,
                metadata_only: false,
            },
            Some(endpoint),
        )?;
        pending.artifact.entry = path.strip_prefix(&root)?.to_owned();
        let lease = ArtifactLease::external(path);
        *downloaded = Some(pending);

        Ok(lease)
    }

    fn local_source_lease(&self, model: &ModelEntry, path: &Path) -> Result<ArtifactLease> {
        let Some(id) = &model.retained_source else {
            return Ok(ArtifactLease::external(path.to_owned()));
        };
        let catalog = self.lock()?;
        let artifact = catalog
            .value
            .artifacts
            .get(id)
            .context("source artifact disappeared")?;

        self.lease(artifact)
    }

    fn export(&self, pending: &Pending, path: &Path) -> Result<Artifact> {
        ensure!(!path.exists(), "--output must name a new directory");

        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));

        crate::storage::create_private_dir(parent)?;
        self.require_external_path(&parent.canonicalize()?)?;
        // Atomically claim the final directory so concurrent exporters cannot mix files.
        std::fs::DirBuilder::new().mode(0o700).create(path)?;
        crate::storage::require_space(path, super::store::bytes(&pending.lease.path)?)?;
        local::copy_store(&pending.lease.path, path, false)?;
        sync_files(path)?;

        Ok(Artifact {
            id: super::id(),
            kind: ArtifactKind::Prepared,
            external: Some(path.canonicalize()?),
            entry: PathBuf::new(),
            ready: true,
        })
    }

    fn discard(&self, id: &str) -> Result<()> {
        let mut catalog = self.lock()?;

        ensure!(
            super::store::references(&catalog.value, id) == 0,
            "cannot discard referenced artifact"
        );

        let lease = self.lease_file(id)?;

        lease
            .try_lock()
            .context("temporary artifact is still in use")?;

        let path = self.artifact_root(id);

        if path.exists() {
            std::fs::remove_dir_all(path)?;
        }

        catalog.value.artifacts.remove(id);

        catalog.save()
    }
}

pub(super) fn check_local(model: &ModelEntry) -> Result<()> {
    if let Source::Local { path, fingerprint } = &model.source {
        ensure!(
            &local::fingerprint(path)? == fingerprint,
            "local source changed; register the changed checkpoint before packing"
        );
    }

    Ok(())
}

fn sync_files(path: &Path) -> Result<()> {
    for entry in std::fs::read_dir(path)? {
        let path = entry?.path();

        if path.is_file() {
            std::fs::File::open(path)?.sync_all()?;
        }
    }

    std::fs::File::open(path)?.sync_all()?;

    Ok(())
}

/// Unlink copied variants before rebuilding: their files may share the old store's inode.
fn remove_variants(path: &Path, experts: &[u32]) -> Result<()> {
    for bits in experts.iter().filter(|&&bits| bits != 4) {
        for name in [format!("experts{bits}.bin"), format!("manifest{bits}.json")] {
            let file = path.join(name);

            if file.exists() {
                std::fs::remove_file(file)?;
            }
        }
    }

    Ok(())
}

/// Make a copied legacy store self-contained before extending its expert variants.
fn copy_runtime_metadata(lease: &ArtifactLease, target: &Path) -> Result<()> {
    let metadata = lease.runtime_path();

    if metadata == lease.path {
        return Ok(());
    }

    for name in ["config.json", "tokenizer.json"] {
        let source = metadata.join(name);
        let destination = target.join(name);

        if source.is_file() && !destination.exists() {
            std::fs::copy(source, destination)?;
        }
    }

    qwen4_exp::pack::copy_chat_metadata(metadata, target)
}
