//! Hardware-independent checkpoint inspection and preparation requirements.
//!
//! Adapters assign meaning to checkpoint names. Consumers use typed roles,
//! encodings, and data references; display names never select execution behavior.

mod format;
pub mod index;
mod inspect;
mod preparation;
pub(crate) mod qwen;

pub use cherenkov_model_data::{
    AffineOffset, BitPacking, ByteSource, DataSpan, Dtype, GgmlEncoding, MappedBytes,
    MappedObjects, ObjectId, ObjectInfo, StoredTensor, Tensor, TensorEncoding, TensorId,
    TensorRole, TensorType,
};
pub use format::*;
pub use inspect::Checkpoint;
pub(crate) use inspect::describe_raw;
pub use preparation::*;
use serde::{Deserialize, Serialize};

/// Architecture-adapted metadata and storage views, independent of execution hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescription {
    /// Version of this description's serialized schema.
    pub schema_version: u32,
    /// Container format and conventions used to interpret its tensors.
    pub format: CheckpointFormat,
    /// Recognized architecture, or its original name when no adapter is available.
    pub architecture: Architecture,
    /// Source metadata retained by the adapter.
    pub metadata: serde_json::Value,
    /// Tensor descriptors, indexed by `TensorId`.
    pub tensors: Vec<Tensor>,
    /// N-gram hashing and shard layout, when present and understood.
    pub ngram: Option<NgramTable>,
}

/// Model architecture recognized by the loader.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Architecture {
    Qwen4Exp,
    Opaque { name: String },
}

/// A table's hashing semantics are separate from its row encoding and placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramTable {
    /// Hash parameters used to select table rows.
    pub hashing: NgramHash,
    /// Row ranges and the tensors storing them.
    pub shards: Vec<TableShard>,
}

/// Hashing scheme and parameters, separate from the table's stored representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NgramHash {
    Qwen4Exp {
        ngram_size: u64,
        heads_per_ngram: u64,
        head_offsets: Vec<u64>,
        head_vocab_sizes: Vec<u64>,
        layer_multipliers: Vec<i64>,
    },
    Opaque {
        name: String,
    },
}

/// A contiguous logical row range backed by a tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableShard {
    /// First row in the logical table.
    pub first_row: u64,
    /// Number of rows in this shard.
    pub rows: u64,
    /// Backing tensor in the enclosing model description.
    pub tensor: TensorId,
}

#[cfg(test)]
#[path = "../../tests/unit/model/mod.rs"]
mod tests;
