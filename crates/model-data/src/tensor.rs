use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// Identifies an object within an opened checkpoint. It is not a filename or
/// a content digest; a store can assign content identities while copying it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObjectId(pub usize);

/// A byte range within one object; offsets are independent of tensor encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSpan {
    /// Object containing this range, relative to its byte source.
    pub object: ObjectId,
    /// Byte offset from the beginning of the object.
    pub offset: u64,
    /// Length of the range in bytes.
    pub length: u64,
}

/// A logical tensor and the physical storage used to represent its values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// Original or adapter-assigned display name.
    pub name: String,
    /// Semantic role assigned by an architecture adapter.
    pub role: TensorRole,
    /// Logical dimensions, absent when an unknown encoding hides them.
    pub logical: Option<TensorType>,
    /// Physical representation and references to its stored bytes.
    pub encoding: TensorEncoding,
}

/// Logical values are distinct from the physical types used to encode them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorType {
    /// Logical dimensions in axis order, before packing or quantization.
    pub shape: Vec<u64>,
    /// Dense tensors declare an element type. Quantized weights do not choose
    /// the execution/accumulation type; a consumer must make that choice.
    pub dtype: Option<Dtype>,
}

/// Index of a tensor in its containing inventory or model description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TensorId(pub usize);

/// An architecture-assigned tensor role, independent of its checkpoint name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    Projection,
    Embedding,
    Expert,
    Router,
    Convolution,
    Norm,
    NgramEmbedding,
    Buffer,
    Opaque,
}

/// Scalar types supported by physical tensor views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    Bool,
    U8,
    I8,
    U16,
    I16,
    I32,
    U64,
    F64,
    U32,
    I64,
    F32,
    F16,
    Bf16,
}

impl Dtype {
    /// Number of bytes occupied by one stored scalar.
    pub fn bytes(self) -> u64 {
        match self {
            Self::I64 | Self::U64 | Self::F64 => 8,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::F16 | Self::Bf16 | Self::U16 | Self::I16 => 2,
            Self::Bool | Self::U8 | Self::I8 => 1,
        }
    }
}

/// Byte strides also describe interleaved records without introducing a
/// table-specific layout or treating packed 4-bit values as addressable bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTensor {
    /// Type of the addressable storage elements, such as packed U32 words.
    pub dtype: Dtype,
    /// Dimensions of the stored elements.
    pub shape: Vec<u64>,
    /// Byte distance between adjacent elements along each axis.
    pub byte_strides: Vec<u64>,
    /// Byte range containing the view, including any gaps between elements.
    pub data: DataSpan,
}

impl StoredTensor {
    /// Construct and validate a row-major view with the last axis contiguous.
    /// Fail on size overflow or when the view exceeds its data span.
    pub fn contiguous(dtype: Dtype, shape: Vec<u64>, data: DataSpan) -> Result<Self> {
        let mut stride = dtype.bytes();
        let mut byte_strides = vec![0; shape.len()];

        for (dim, out) in shape.iter().zip(&mut byte_strides).rev() {
            *out = stride;
            stride = stride
                .checked_mul(*dim)
                .ok_or_else(|| anyhow::anyhow!("tensor size overflow"))?;
        }

        let tensor = Self {
            dtype,
            shape,
            byte_strides,
            data,
        };

        tensor.validate()?;

        Ok(tensor)
    }

    /// Check stride rank, arithmetic overflow, and extent within the data span.
    /// The backing object's bounds are checked separately by the byte source.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.shape.len() == self.byte_strides.len(),
            "tensor stride rank mismatch"
        );

        ensure!(
            self.data.offset.checked_add(self.data.length).is_some(),
            "data span overflow"
        );

        if self.shape.contains(&0) {
            return Ok(());
        }

        let mut extent = self.dtype.bytes();

        for (&dim, &stride) in self.shape.iter().zip(&self.byte_strides) {
            let last = (dim - 1)
                .checked_mul(stride)
                .ok_or_else(|| anyhow::anyhow!("tensor stride overflow"))?;
            extent = extent
                .checked_add(last)
                .ok_or_else(|| anyhow::anyhow!("tensor extent overflow"))?;
        }

        ensure!(
            extent <= self.data.length,
            "tensor view exceeds its data span"
        );

        Ok(())
    }
}

/// Storage layout and quantization metadata, with explicit references to all data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TensorEncoding {
    /// Addressable scalar values with explicit shape and strides.
    Dense {
        /// Physical view of the scalar values.
        tensor: StoredTensor,
    },
    /// Grouped integer codes reconstructed using scales and an affine offset.
    Affine {
        /// Bits per quantized value.
        bits: u8,
        /// Logical values sharing one scale and offset.
        group_size: u64,
        /// Axis along which quantization groups are formed.
        group_axis: usize,
        /// Order of codes within each storage word.
        packing: BitPacking,
        /// Packed integer codes.
        codes: StoredTensor,
        /// One scale per quantization group.
        scales: StoredTensor,
        /// Additive biases or integer zero points, one per group.
        offset: AffineOffset,
    },
    /// GGML blocks whose layout is determined by the encoding code.
    Ggml {
        encoding: GgmlEncoding,
        data: DataSpan,
    },
    /// Uninterpreted storage retained for a future adapter.
    Opaque {
        name: String,
        metadata: serde_json::Value,
        data: Vec<DataSpan>,
    },
}

/// Placement of quantized codes within addressable storage words.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BitPacking {
    /// Consecutive codes occupy a U32 word from least to most significant bits.
    LowFirstU32,
}

/// Per-group offset used to reconstruct affine-quantized values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AffineOffset {
    /// Reconstruction uses `code * scale + bias`.
    Bias { tensor: StoredTensor },
    /// Reconstruction uses `(code - zero_point) * scale`.
    ZeroPoint { tensor: StoredTensor },
}

/// Recognized GGML element and block encodings; unknown numeric codes are preserved.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GgmlEncoding {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
    Bf16,
    Opaque { code: u32 },
}

impl TensorEncoding {
    /// Explicit references remain enumerable even for an unknown encoding.
    pub fn data(&self) -> Vec<&DataSpan> {
        match self {
            Self::Dense { tensor } => vec![&tensor.data],
            Self::Affine {
                codes,
                scales,
                offset,
                ..
            } => {
                let (AffineOffset::Bias { tensor } | AffineOffset::ZeroPoint { tensor }) = offset;

                vec![&codes.data, &scales.data, &tensor.data]
            }
            Self::Ggml { data, .. } => vec![data],
            Self::Opaque { data, .. } => data.iter().collect(),
        }
    }
}

impl Tensor {
    /// Build a descriptor, inferring a logical dtype only for dense storage.
    /// Call [`Self::validate`] to check consistency with the physical encoding.
    pub fn new(
        name: String,
        role: TensorRole,
        shape: Option<Vec<u64>>,
        encoding: TensorEncoding,
    ) -> Self {
        let dtype = match &encoding {
            TensorEncoding::Dense { tensor } => Some(tensor.dtype),
            _ => None,
        };

        Self {
            name,
            role,
            logical: shape.map(|shape| TensorType { shape, dtype }),
            encoding,
        }
    }

    /// Return logical dimensions, if the encoding's shape is known.
    pub fn shape(&self) -> Option<&[u64]> {
        self.logical.as_ref().map(|t| t.shape.as_slice())
    }

    /// Check the encoding and its agreement with the logical shape and dtype.
    pub fn validate(&self) -> Result<()> {
        self.encoding.validate()?;

        if let TensorEncoding::Dense { tensor } = &self.encoding {
            ensure!(
                self.shape() == Some(tensor.shape.as_slice()),
                "dense logical shape mismatch"
            );
            ensure!(
                self.logical.as_ref().and_then(|t| t.dtype) == Some(tensor.dtype),
                "dense logical dtype mismatch"
            );
        }

        if let TensorEncoding::Affine {
            codes,
            bits,
            group_axis,
            ..
        } = &self.encoding
        {
            let mut shape = codes.shape.clone();
            shape[*group_axis] = shape[*group_axis]
                .checked_mul(32 / u64::from(*bits))
                .ok_or_else(|| anyhow::anyhow!("affine shape overflow"))?;

            ensure!(
                self.shape() == Some(shape.as_slice()),
                "affine logical shape mismatch"
            );
        }

        Ok(())
    }
}

impl TensorEncoding {
    /// Check component layouts and supported affine grouping constraints.
    /// For GGML and opaque data, only span arithmetic is checked.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Dense { tensor } => tensor.validate(),
            Self::Affine {
                bits,
                group_size,
                group_axis,
                codes,
                scales,
                offset,
                ..
            } => {
                let (AffineOffset::Bias { tensor: offsets }
                | AffineOffset::ZeroPoint { tensor: offsets }) = offset;

                codes.validate()?;
                scales.validate()?;
                offsets.validate()?;
                ensure!(
                    matches!(bits, 2 | 4 | 8) && codes.dtype == Dtype::U32,
                    "unsupported affine word packing"
                );
                ensure!(
                    codes.shape.len().checked_sub(1) == Some(*group_axis),
                    "affine groups must span the final axis"
                );
                ensure!(
                    scales.shape.len() == codes.shape.len() && offsets.shape == scales.shape,
                    "affine component rank mismatch"
                );
                ensure!(
                    scales.shape[..*group_axis] == codes.shape[..*group_axis],
                    "affine component row mismatch"
                );

                let width = codes.shape[*group_axis]
                    .checked_mul(32 / u64::from(*bits))
                    .ok_or_else(|| anyhow::anyhow!("affine shape overflow"))?;

                ensure!(
                    *group_size > 0 && width > 0 && width.is_multiple_of(*group_size),
                    "invalid affine group size"
                );
                ensure!(
                    scales.shape[*group_axis] == width / *group_size,
                    "affine scale count mismatch"
                );

                Ok(())
            }
            Self::Ggml { data, .. } => check_span(data),
            Self::Opaque { data, .. } => data.iter().try_for_each(check_span),
        }
    }
}

fn check_span(span: &DataSpan) -> Result<()> {
    ensure!(
        span.offset.checked_add(span.length).is_some(),
        "data span overflow"
    );

    Ok(())
}
