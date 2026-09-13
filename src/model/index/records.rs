use crate::model::ModelDescription;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

/// Catalog identity, source lineage, and selected artifacts for one model.
#[derive(Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Stable catalog ID accepted directly as a model selector.
    pub id: String,
    /// Optional unique alias accepted directly as a model selector.
    pub name: Option<String>,
    /// Original source location and the identity recorded at registration.
    pub source: Source,
    /// Source description captured at registration.
    pub description: ModelDescription,
    /// Selected prepared artifact ID, if preparation has been published.
    pub prepared: Option<String>,
    /// Retained source artifact ID, if the index tracks a local source copy.
    pub retained_source: Option<String>,
}

/// Source identity used to recognize repeat registration and verify local imports.
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Source {
    /// Local checkpoint inspected at registration.
    Local {
        /// Canonical source path.
        path: PathBuf,
        /// File metadata and configuration fingerprint, not a digest of all weights.
        fingerprint: String,
    },
    /// HF repository pinned to an immutable commit.
    HuggingFace {
        /// Repository identifier in `owner/name` form.
        repo: String,
        /// Full commit hash resolved during registration.
        revision: String,
        /// Hub endpoint used to inspect and later download this source.
        endpoint: String,
    },
}

/// An artifact's role in a model's preparation lifecycle.
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// Weights and metadata prepared for the inference engine.
    Prepared,
    /// Source checkpoint retained for future preparation.
    Source,
}

/// Catalog record for an owned directory or a caller-owned external location.
#[derive(Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Stable ID used by model references and artifact lease locks.
    pub id: String,
    /// Prepared output or retained source.
    pub kind: ArtifactKind,
    /// An external location is never owned, even when below the configured root.
    pub external: Option<PathBuf>,
    /// Relative to the owned artifact directory; HF snapshots have nested paths.
    pub entry: PathBuf,
    /// Whether preparation or acquisition completed before publication.
    pub ready: bool,
}

#[derive(Serialize, Deserialize)]
pub(super) struct Catalog {
    pub version: u32,
    pub models: BTreeMap<String, ModelEntry>,
    pub artifacts: BTreeMap<String, Artifact>,
    #[serde(default)]
    pub stores: BTreeMap<String, super::DiskStore>,
}

impl Default for Catalog {
    fn default() -> Self {
        Self {
            version: 1,
            models: BTreeMap::new(),
            artifacts: BTreeMap::new(),
            stores: BTreeMap::new(),
        }
    }
}

/// Current availability and storage totals for a registered model.
#[derive(Serialize)]
pub struct ModelSummary {
    /// Stable catalog ID.
    pub id: String,
    /// Resolvable selector: alias, full source reference, or ID when ambiguous.
    pub reference: String,
    /// Optional model alias.
    pub name: Option<String>,
    /// Architecture recorded in the source description.
    pub architecture: crate::model::Architecture,
    /// Original source identity.
    pub source: Source,
    /// Whether the local source path or a retained HF source currently exists.
    pub source_local: bool,
    /// Whether the referenced prepared artifact path exists.
    pub prepared: bool,
    /// Expert bit widths reported by [`super::prepared_precisions`], descending.
    pub precisions: Vec<u32>,
    /// Referenced file lengths, not filesystem blocks reclaimed by removal.
    pub owned_bytes: u64,
    /// File lengths in referenced external artifacts; excludes untracked source files.
    pub external_bytes: u64,
}

/// Model summary with preparation requirements and artifact ownership details.
#[derive(Serialize)]
pub struct ModelDetails {
    /// Availability and byte totals, flattened in serialized output.
    #[serde(flatten)]
    pub summary: ModelSummary,
    /// Direct when a prepared path exists; otherwise the source's requirements.
    pub preparation: crate::model::Preparation,
    /// Artifacts referenced by this model.
    pub artifacts: Vec<ArtifactDetails>,
}

/// Location, ownership, and reference count of a catalog artifact.
#[derive(Serialize)]
pub struct ArtifactDetails {
    /// Stable artifact ID.
    pub id: String,
    /// Prepared output or retained source.
    pub kind: ArtifactKind,
    /// Resolved artifact entry path.
    pub path: PathBuf,
    /// Whether garbage collection may delete the artifact's files.
    pub owned: bool,
    /// Whether the entry path currently exists; not a full integrity check.
    pub available: bool,
    /// Sum of file lengths, including shared hard links at each referenced path.
    pub bytes: u64,
    /// Number of model artifact references, excluding live leases.
    pub references: usize,
}

/// Garbage collection candidates and artifacts skipped because they are leased.
#[derive(Default, Serialize)]
pub struct GcReport {
    /// Whether collection only inspected candidates.
    pub dry_run: bool,
    /// Artifact IDs removed, or eligible for removal during a dry run.
    pub artifacts: Vec<String>,
    /// Unreferenced artifact IDs skipped because their lease lock was unavailable.
    pub leased: Vec<String>,
    /// Owned candidate file lengths; shared hard links can reduce reclaimed space.
    pub candidate_bytes: u64,
}

/// References released by removal; their files have not been collected yet.
#[derive(Serialize)]
pub struct Removal {
    /// Model ID whose references changed.
    pub id: String,
    /// Whether only the retained source reference was released.
    pub source_only: bool,
    /// Released artifact IDs, which may still be referenced by other models.
    pub released_artifacts: Vec<String>,
}
