use serde::{Deserialize, Serialize};

/// Container layout and the conventions used to interpret tensor names and encodings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointFormat {
    /// Physical container format.
    pub container: ContainerFormat,
    /// Naming and quantization conventions recognized by an adapter.
    pub conventions: CheckpointConventions,
}

/// On-disk container family, including Cherenkov's prepared format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContainerFormat {
    Safetensors,
    Gguf { version: u32 },
    CherenkovPacked { version: u32 },
    Opaque { name: String },
}

/// Recognized naming and encoding conventions within a container.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CheckpointConventions {
    Qwen4ExpHuggingFace,
    Qwen4ExpMlx,
    MlxAffine,
    Qwen4ExpCherenkov,
    Gguf,
    Opaque { name: String },
}
