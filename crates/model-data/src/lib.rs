//! Container metadata and lazy byte access, independent of model execution.
//!
//! Readers preserve names and metadata. Architecture adapters assign tensor
//! roles and interpret quantization conventions after reading the container.

pub mod discovery;
mod gguf;
pub mod mlx;
mod safetensor;
pub use safetensor::read as read_safetensors;
mod source;
mod tensor;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
pub use source::{ByteSource, MappedBytes, MappedObjects, ObjectInfo};
use std::path::Path;
pub use tensor::*;

/// Container format identified before applying model-specific conventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContainerFormat {
    Safetensors,
    Gguf { version: u32 },
    Opaque { name: String, version: Option<u32> },
}

/// Container metadata and tensor descriptors whose bytes remain in a separate source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inventory {
    /// Container format and any recognized version.
    pub format: ContainerFormat,
    /// Container metadata, including configuration when available.
    pub metadata: serde_json::Value,
    /// Tensor descriptors whose object IDs refer to the accompanying source.
    pub tensors: Vec<Tensor>,
}

/// A local container inventory with the mappings backing its tensor data.
pub struct Checkpoint {
    /// Metadata read from the container, before architecture adaptation.
    pub inventory: Inventory,
    /// Immutable backing files for the inventory's data spans.
    pub objects: MappedObjects,
}

impl Checkpoint {
    /// Open a safetensors folder/file or a GGUF file. Files must remain
    /// unchanged while the checkpoint or any mapped view is alive.
    pub fn open(path: &Path) -> Result<Self> {
        if path.is_dir() {
            return safetensor::open(path);
        }

        let objects = MappedObjects::open([path])?;
        let bytes = objects.mapping(ObjectId(0))?;
        let inventory = if bytes.starts_with(b"GGUF") {
            gguf::inspect(bytes)?
        } else {
            safetensor::read(&objects, ObjectId(0))?
        };
        let checkpoint = Self { inventory, objects };

        checkpoint.validate()?;

        Ok(checkpoint)
    }

    /// Check unique tensor names, tensor layouts, and bounds within backing files.
    /// This does not verify tensor values or architecture compatibility.
    pub fn validate(&self) -> Result<()> {
        let mut names = std::collections::HashSet::new();

        for tensor in &self.inventory.tensors {
            ensure!(
                names.insert(&tensor.name),
                "duplicate tensor {}",
                tensor.name
            );
            tensor.validate()?;

            for span in tensor.encoding.data() {
                self.objects.bytes(span)?;
            }
        }

        Ok(())
    }
}
