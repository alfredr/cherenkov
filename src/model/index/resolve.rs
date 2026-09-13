//! A single resolver registers explicit sources and reuses pinned index entries.

use super::{
    ModelDetails, ModelIndex, disk, reference,
    selector::{self, Selector},
};
use anyhow::{Context, Result, ensure};
use std::path::Path;

/// Optional registration settings shared by prepare and explicit model registration.
#[derive(Default)]
pub struct ResolveOptions<'a> {
    /// Optional alias; an existing alias is never silently replaced.
    pub name: Option<&'a str>,
    /// HF revision override; mutually exclusive with an @revision suffix.
    pub revision: Option<&'a str>,
    /// Optional credential used only if remote metadata must be fetched.
    pub token: Option<&'a str>,
}

impl ResolveOptions<'_> {
    /// Validate source-specific options before catalog lookup or remote access.
    fn apply(&self, selector: &mut Selector) -> Result<()> {
        match selector {
            Selector::Hub { revision, .. } => {
                ensure!(
                    revision.is_none() || self.revision.is_none(),
                    "specify the revision with @revision or --revision, not both"
                );

                if let Some(value) = self.revision {
                    selector::validate_revision(value)?;

                    *revision = Some(value.to_owned());
                }
            }
            Selector::Path(_) | Selector::Disk { .. } => {
                ensure!(
                    self.revision.is_none() && self.token.is_none(),
                    "revision and token require an HF source"
                );
            }
            Selector::Registered(_) | Selector::LocalRevision { .. } => {
                ensure!(self.revision.is_none(), "revision requires a source URI");
            }
        }

        Ok(())
    }
}

impl ModelIndex {
    /// Resolve an alias, ID, URI, or explicit path; register unknown sources.
    /// Unqualified known HF URIs reuse their pinned entry without contacting HF.
    pub fn select(&self, input: &Path, options: ResolveOptions<'_>) -> Result<ModelDetails> {
        super::validate_name(options.name)?;

        let mut selector = selector::parse(input)?;

        options.apply(&mut selector)?;

        // Explicit paths must be inspected again to detect changed local sources.
        if let Selector::Path(path) = &selector {
            return self.add_local(path, options.name);
        }

        let input = input.to_str().context("invalid model selector")?;
        let reference = match options.revision {
            Some(revision) => format!("{input}@{revision}"),
            None => input.to_owned(),
        };
        let existing = reference::find(&self.lock()?.value, &selector, &reference)?.cloned();

        if let Some(model) = existing {
            return self.name_existing(model, options.name);
        }

        match selector {
            Selector::Hub { repo, revision } => {
                let (source, description) = super::hub::inspect(
                    &repo,
                    revision.as_deref().unwrap_or("main"),
                    options.token,
                )?;
                let id = self.register(source, description, options.name, None)?;

                self.show(&id)
            }
            Selector::Disk {
                store,
                repo,
                revision,
            } => self.select_disk(&store, &repo, revision.as_deref(), options.name),
            _ => anyhow::bail!("model {reference:?} is not registered"),
        }
    }

    /// Locate disk files only after lookup has ruled out an existing indexed identity.
    fn select_disk(
        &self,
        name: &str,
        repo: &str,
        revision: Option<&str>,
        alias: Option<&str>,
    ) -> Result<ModelDetails> {
        let store = disk::registered(&self.lock()?.value, name)?.clone();

        let path = store.locate(repo, revision)?;

        if matches!(store.layout, super::DiskLayout::Directory)
            && let Some(revision) = revision
        {
            ensure!(
                reference::is_revision(revision)
                    && super::local::fingerprint(&path)?.starts_with(revision),
                "disk model does not match the requested revision"
            );
        }

        self.add_local(&path, alias)
    }

    /// Assign an optional name through the normal collision and ownership checks.
    fn name_existing(&self, model: super::ModelEntry, name: Option<&str>) -> Result<ModelDetails> {
        if name.is_none() {
            return self.show(&model.id);
        }

        let id = self.register(model.source, model.description, name, None)?;

        self.show(&id)
    }
}
