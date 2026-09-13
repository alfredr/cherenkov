use super::*;
use serde::{Deserialize, Serialize};

/// Representation requirements for loading with the current Qwen4-exp engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Preparation {
    /// The described representation can be consumed without conversion.
    Direct,
    /// Supported after the listed preparation steps.
    Required { steps: Vec<PreparationStep> },
    /// The current engine cannot prepare this representation.
    Unsupported { reasons: Vec<CompatibilityIssue> },
}

/// A transformation needed to produce the engine's prepared representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparationStep {
    /// Normalize native Qwen4-exp tensor names and layout.
    NormalizeQwen4Exp,
    /// Convert supported dense weights to affine 4-bit storage.
    QuantizeAffineQ4,
    /// Group expert weights into records for streaming and residency.
    PackExpertRecords,
    /// Interleave n-gram codes, scales, and biases by row.
    InterleaveNgramRows,
}

/// A representation constraint that prevents preparation by the current engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CompatibilityIssue {
    Architecture,
    Conventions,
    GgufConversion,
    Encoding { tensor: String },
    GroupSize { tensor: String, group_size: u64 },
    Shape { tensor: String },
}

impl ModelDescription {
    /// Requirements for the current Cherenkov Qwen4-exp packed format. This
    /// checks representation compatibility, not hardware or model completeness.
    pub fn preparation(&self) -> Preparation {
        if matches!(self.format.container, ContainerFormat::Gguf { .. }) {
            return Preparation::Unsupported {
                reasons: vec![CompatibilityIssue::GgufConversion],
            };
        }

        if !matches!(self.architecture, Architecture::Qwen4Exp) {
            return Preparation::Unsupported {
                reasons: vec![CompatibilityIssue::Architecture],
            };
        }

        let native = matches!(
            self.format.conventions,
            CheckpointConventions::Qwen4ExpHuggingFace
        );
        let known = native
            || matches!(
                self.format.conventions,
                CheckpointConventions::Qwen4ExpMlx | CheckpointConventions::Qwen4ExpCherenkov
            );

        if !known {
            return Preparation::Unsupported {
                reasons: vec![CompatibilityIssue::Conventions],
            };
        }

        let reasons: Vec<_> = self
            .tensors
            .iter()
            .filter_map(|t| compatibility(t, native))
            .collect();

        if !reasons.is_empty() {
            return Preparation::Unsupported { reasons };
        }

        if matches!(
            self.format.container,
            ContainerFormat::CherenkovPacked { version: 1 }
        ) {
            return Preparation::Direct;
        }

        let mut steps = Vec::new();

        if native {
            steps.extend([
                PreparationStep::NormalizeQwen4Exp,
                PreparationStep::QuantizeAffineQ4,
            ]);
        }

        steps.extend([
            PreparationStep::PackExpertRecords,
            PreparationStep::InterleaveNgramRows,
        ]);

        Preparation::Required { steps }
    }
}

fn compatibility(tensor: &Tensor, native: bool) -> Option<CompatibilityIssue> {
    let quantized_role = matches!(
        tensor.role,
        TensorRole::Projection
            | TensorRole::Embedding
            | TensorRole::Expert
            | TensorRole::NgramEmbedding
    );

    if !quantized_role {
        return None;
    }

    let width = tensor.shape().and_then(|s| s.last()).copied().unwrap_or(0);
    let alignment = if tensor.role == TensorRole::NgramEmbedding {
        8
    } else {
        64
    };

    if width == 0 || !width.is_multiple_of(alignment) {
        return Some(CompatibilityIssue::Shape {
            tensor: tensor.name.clone(),
        });
    }

    match &tensor.encoding {
        TensorEncoding::Dense { tensor } if native && tensor.dtype == Dtype::Bf16 => None,
        TensorEncoding::Affine {
            bits: 4,
            group_size,
            scales,
            offset: AffineOffset::Bias { tensor: biases },
            ..
        } if scales.dtype == Dtype::Bf16 && biases.dtype == Dtype::Bf16 => {
            let valid = if tensor.role == TensorRole::NgramEmbedding {
                matches!(group_size, 8 | 16 | 32 | 64)
            } else {
                *group_size == 64
            };

            (!valid).then(|| CompatibilityIssue::GroupSize {
                tensor: tensor.name.clone(),
                group_size: *group_size,
            })
        }
        _ => Some(CompatibilityIssue::Encoding {
            tensor: tensor.name.clone(),
        }),
    }
}
