use anyhow::{Result, bail};
use cherenkov_model_data::{Checkpoint, Dtype as StoredDtype, ObjectId, TensorEncoding};
use std::{collections::HashMap, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    U32,
    F32,
    F16,
    BF16,
    I64,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Dtype::I64 => 8,
            Dtype::U32 | Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Object index within the source checkpoint.
    pub shard: usize,
    /// Absolute byte offset of the tensor data within the shard file.
    pub offset: usize,
    pub nbytes: usize,
}

pub struct ModelWeights {
    pub checkpoint: Checkpoint,
    pub tensors: HashMap<String, TensorInfo>,
}

impl ModelWeights {
    /// Keep the container's mappings; packing reads bytes without realigning
    /// or duplicating the checkpoint in memory.
    pub fn load_raw(path: &Path) -> Result<Self> {
        let checkpoint = Checkpoint::open(path)?;
        let mut tensors = HashMap::new();

        for tensor in &checkpoint.inventory.tensors {
            let TensorEncoding::Dense { tensor: stored } = &tensor.encoding else {
                bail!(
                    "{}: packer requires a supported dense storage type",
                    tensor.name
                );
            };
            let dtype = match stored.dtype {
                StoredDtype::U32 => Dtype::U32,
                StoredDtype::F32 => Dtype::F32,
                StoredDtype::F16 => Dtype::F16,
                StoredDtype::Bf16 => Dtype::BF16,
                StoredDtype::I64 => Dtype::I64,
                other => bail!("{}: packer does not support {other:?}", tensor.name),
            };

            tensors.insert(
                tensor.name.clone(),
                TensorInfo {
                    dtype,
                    shape: stored
                        .shape
                        .iter()
                        .map(|&n| usize::try_from(n))
                        .collect::<Result<_, _>>()?,
                    shard: stored.data.object.0,
                    offset: usize::try_from(stored.data.offset)?,
                    nbytes: usize::try_from(stored.data.length)?,
                },
            );
        }

        Ok(Self {
            checkpoint,
            tensors,
        })
    }

    pub fn tensor_bytes(&self, info: &TensorInfo) -> &[u8] {
        // TensorInfo comes from validated container spans in load_raw.
        let map = self
            .checkpoint
            .objects
            .mapping(ObjectId(info.shard))
            .expect("validated object");

        &map[info.offset..info.offset + info.nbytes]
    }
}
