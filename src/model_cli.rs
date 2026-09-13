//! Model-index commands resolve to library operations and shared report views.
use anyhow::Result;
use cherenkov::{
    model::index::{DiskLayout, ModelIndex, ResolveOptions},
    storage::Paths,
};
use clap::Subcommand;
use std::path::Path;

#[derive(Subcommand)]
pub(crate) enum Action {
    /// Register a local checkpoint or HF metadata without downloading weights
    Add {
        source: String,
        /// Optional alias; source references work without one
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        revision: Option<String>,
        #[arg(long)]
        hf_token: Option<String>,
    },
    /// List indexed models without contacting remote sources
    List,
    /// Alias for inspect
    #[command(hide = true)]
    Show { reference: String },
    /// Release index references; use gc to delete unreferenced owned stores
    Remove {
        reference: String,
        #[arg(long)]
        source_only: bool,
    },
    /// Delete unreferenced owned stores; live readers remain protected
    Gc {
        #[arg(long)]
        dry_run: bool,
    },
}

pub(crate) fn run(paths: Paths, action: Action, json: bool) -> Result<()> {
    let index = ModelIndex::new(paths);

    match action {
        Action::Add {
            source,
            name,
            revision,
            hf_token,
        } => {
            let result = index.add(
                &source,
                revision.as_deref(),
                name.as_deref(),
                hf_token.as_deref(),
            )?;

            crate::cli_output::models::show(&result, json)
        }
        Action::List => crate::cli_output::models::list(&index.list()?, json),
        Action::Show { reference } => {
            let selected = index.show(&reference)?;
            let description = index.description(&reference)?;

            crate::cli_output::models::inspect_indexed(&selected, &description, json)
        }
        Action::Remove {
            reference,
            source_only,
        } => crate::cli_output::models::removed(&index.remove(&reference, source_only)?, json),
        Action::Gc { dry_run } => crate::cli_output::models::gc(&index.gc(dry_run)?, json),
    }
}

/// Resolve every source through the index before preparing selected variants.
pub(crate) fn prepare(
    paths: Paths,
    input: &Path,
    options: ResolveOptions<'_>,
    output: Option<&Path>,
    experts: &[u32],
    keep_source: bool,
) -> Result<()> {
    let index = ModelIndex::new(paths);
    let token = options.token;
    let selected = index.select(input, options)?;
    let result = index.pack(
        &selected.summary.id,
        cherenkov::model::index::PackOptions {
            output,
            experts,
            keep_source,
            token,
            repack: false,
        },
    )?;

    crate::cli_output::models::show(&result, false)
}

/// Inspect a registered artifact or register an explicit source before inspection.
pub(crate) fn inspect(paths: Paths, input: &Path, json: bool) -> Result<()> {
    let index = ModelIndex::new(paths);
    let selected = index.select(input, ResolveOptions::default())?;
    let description = index.description(&selected.summary.id)?;

    crate::cli_output::models::inspect_indexed(&selected, &description, json)
}

/// Registration and availability controls for externally owned disk stores.
#[derive(Subcommand)]
pub(crate) enum StoreAction {
    /// Register a filesystem root without importing its models
    Add {
        name: String,
        path: std::path::PathBuf,
        #[arg(long, value_enum, default_value = "directory")]
        layout: DiskLayout,
    },
    /// List registered stores
    List,
    /// Forget a store without deleting files or indexed models
    Remove { name: String },
    /// Enable source resolution through a store
    Enable { name: String },
    /// Disable source resolution through a store
    Disable { name: String },
}

/// Execute store management through the same persistent index as model commands.
pub(crate) fn store(paths: Paths, action: StoreAction, json: bool) -> Result<()> {
    let index = ModelIndex::new(paths);

    match action {
        StoreAction::Add { name, path, layout } => {
            index.add_disk_store(&name, &path, layout)?;
        }
        StoreAction::List => {}
        StoreAction::Remove { name } => index.remove_disk_store(&name)?,
        StoreAction::Enable { name } => index.set_disk_store_enabled(&name, true)?,
        StoreAction::Disable { name } => index.set_disk_store_enabled(&name, false)?,
    }

    crate::cli_output::models::stores(&index.disk_stores()?, json)
}
