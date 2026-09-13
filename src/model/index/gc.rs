use super::{GcReport, ModelIndex, Removal, lookup, store};
use anyhow::{Context, Result, ensure};

impl ModelIndex {
    /// Remove a model registration, or release only its retained source reference.
    /// Source-only removal first checks the prepared checkpoint. Files remain until
    /// [`Self::gc`] collects unreferenced owned artifacts; external files are untouched.
    pub fn remove(&self, reference: &str, source_only: bool) -> Result<Removal> {
        let mut catalog = self.lock()?;
        let model = lookup(&catalog.value, reference)?.clone();
        let mut released = Vec::new();

        if source_only {
            let prepared = model
                .prepared
                .as_ref()
                .context("prepare the model before releasing source copies")?;
            let artifact = &catalog.value.artifacts[prepared];

            ensure!(artifact.ready, "prepared artifact is incomplete");

            crate::model::Checkpoint::open(&self.artifact_path(artifact))
                .context("validate the prepared model before releasing its source")?;

            if let Some(id) = model.retained_source {
                released.push(id);
            }

            catalog
                .value
                .models
                .get_mut(&model.id)
                .unwrap()
                .retained_source = None;
        } else {
            released.extend(model.prepared);
            released.extend(model.retained_source);
            catalog.value.models.remove(&model.id);
        }

        catalog.save()?;

        Ok(Removal {
            id: model.id,
            source_only,
            released_artifacts: released,
        })
    }

    /// Collect unreferenced catalog artifacts that have no active lease.
    /// Owned directories are deleted; external records are removed without deleting
    /// their files. A dry run reports candidates without removing either.
    pub fn gc(&self, dry_run: bool) -> Result<GcReport> {
        if !self.paths.data.join("index.json").exists() {
            return Ok(GcReport {
                dry_run,
                ..Default::default()
            });
        }

        let mut catalog = self.lock()?;
        let candidates: Vec<_> = catalog
            .value
            .artifacts
            .values()
            .filter(|a| store::references(&catalog.value, &a.id) == 0)
            .cloned()
            .collect();
        let mut report = GcReport {
            dry_run,
            ..Default::default()
        };

        for artifact in candidates {
            let lease = self.lease_file(&artifact.id)?;

            if lease.try_lock().is_err() {
                report.leased.push(artifact.id);

                continue;
            }

            if artifact.external.is_none() {
                let path = self.artifact_root(&artifact.id);
                report.candidate_bytes += store::bytes(&path)?;

                if !dry_run && path.exists() {
                    std::fs::remove_dir_all(path)?;
                }
            }

            report.artifacts.push(artifact.id.clone());

            if !dry_run {
                catalog.value.artifacts.remove(&artifact.id);
            }
        }

        if !dry_run {
            catalog.save()?;
        }

        Ok(report)
    }
}
