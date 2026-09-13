use anyhow::{Result, ensure};
use cherenkov_model_data::*;
use serde_json::json;
use std::{cell::RefCell, fs, io::Write};

struct MemorySource {
    bytes: Vec<u8>,
    reads: RefCell<Vec<(u64, u64)>>,
}

impl MemorySource {
    fn new(header: serde_json::Value, data: &[u8]) -> Self {
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();

        bytes.extend(header);
        bytes.extend(data);

        Self {
            bytes,
            reads: RefCell::new(Vec::new()),
        }
    }
}

impl ByteSource for MemorySource {
    fn objects(&self) -> Vec<ObjectInfo> {
        vec![ObjectInfo {
            id: ObjectId(0),
            bytes: self.bytes.len() as u64,
        }]
    }

    fn read(&self, span: &DataSpan, output: &mut dyn Write) -> Result<()> {
        ensure!(span.object == ObjectId(0), "unknown object");
        self.reads.borrow_mut().push((span.offset, span.length));
        output
            .write_all(&self.bytes[span.offset as usize..(span.offset + span.length) as usize])?;

        Ok(())
    }
}

#[test]
fn header_inspection_needs_neither_mapping_nor_weight_reads() {
    let source = MemorySource::new(
        json!({
            "__metadata__": {"producer": "audio-test"},
            "encoder.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "codec": {"dtype": "U8", "shape": [3], "data_offsets": [8, 11]},
            "fp8": {"dtype": "F8_E4M3", "shape": [2], "data_offsets": [11, 13]}
        }),
        &[0; 13],
    );
    let inventory = read_safetensors(&source, ObjectId(0)).unwrap();

    assert_eq!(inventory.tensors.len(), 3);
    assert_eq!(inventory.metadata["shards"][0]["producer"], "audio-test");

    let fp8 = inventory.tensors.iter().find(|t| t.name == "fp8").unwrap();

    assert!(matches!(&fp8.encoding, TensorEncoding::Opaque { name, .. } if name == "F8_E4M3"));
    assert_eq!(fp8.shape(), Some([2].as_slice()));
    assert!(source.map(fp8.encoding.data()[0]).unwrap().is_none());

    let reads = source.reads.borrow();

    assert_eq!(reads.len(), 2);
    assert_eq!(reads[1].0 + reads[1].1, (source.bytes.len() - 13) as u64);
}

#[test]
fn mappings_survive_source_drop_and_reject_bad_spans() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bytes");

    fs::write(&path, [1, 2, 3, 4]).unwrap();

    let source = MappedObjects::open([path.as_path()]).unwrap();
    let span = DataSpan {
        object: ObjectId(0),
        offset: 1,
        length: 2,
    };
    let view = source.view(&span).unwrap();
    let mut copied = Vec::new();

    source.read(&span, &mut copied).unwrap();
    assert_eq!(copied, [2, 3]);

    for (id, offset, length) in [(0, u64::MAX, 2), (0, 3, 2), (1, 0, 1)] {
        assert!(
            source
                .view(&DataSpan {
                    object: ObjectId(id),
                    offset,
                    length
                })
                .is_err()
        );
    }

    drop(source);
    assert_eq!(view.as_ref(), &[2, 3]);
}

#[test]
fn malformed_shapes_offsets_and_lengths_are_rejected() {
    for header in [
        json!({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 7]}}),
        json!({"w": {"dtype": "F32", "shape": [2], "data_offsets": [8, 0]}}),
        json!({"w": {"dtype": "F32", "shape": [u64::MAX, 2], "data_offsets": [0, 8]}}),
        json!({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
               "overlap": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}),
    ] {
        let source = MemorySource::new(header, &[0; 8]);

        assert!(read_safetensors(&source, ObjectId(0)).is_err());
    }

    let mut source = MemorySource::new(
        json!({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}),
        &[0; 8],
    );

    source.bytes.pop();
    assert!(read_safetensors(&source, ObjectId(0)).is_err());
    source.bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(read_safetensors(&source, ObjectId(0)).is_err());
}

#[test]
fn folder_keeps_config_index_and_shard_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let source = MemorySource::new(
        json!({"__metadata__": {"custom": "value"}, "w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}),
        &[0; 4],
    );

    fs::write(dir.path().join("part.safetensors"), &source.bytes).unwrap();
    fs::write(
        dir.path().join("config.json"),
        r#"{"model_type":"audio_encoder","custom":{"block":7}}"#,
    )
    .unwrap();

    let index = json!({"metadata":{"total_size":4},"weight_map":{"w":"part.safetensors"}});

    fs::write(
        dir.path().join("model.safetensors.index.json"),
        index.to_string(),
    )
    .unwrap();

    let checkpoint = Checkpoint::open(dir.path()).unwrap();

    assert_eq!(
        checkpoint.inventory.metadata["config"]["custom"]["block"],
        7
    );
    assert_eq!(checkpoint.inventory.metadata["index"], index);
    assert_eq!(
        checkpoint.inventory.metadata["shards"][0]["custom"],
        "value"
    );
    assert_eq!(checkpoint.inventory.tensors[0].role, TensorRole::Opaque);
}

#[test]
fn index_cannot_escape_folder_or_silently_misname_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let source = MemorySource::new(
        json!({"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}),
        &[0; 4],
    );

    fs::write(dir.path().join("part.safetensors"), source.bytes).unwrap();

    for map in [
        json!({"w":"../part.safetensors"}),
        json!({"other":"part.safetensors"}),
        json!({"w":"part.safetensors","missing":"part.safetensors"}),
        json!({}),
    ] {
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            json!({"weight_map":map}).to_string(),
        )
        .unwrap();
        assert!(Checkpoint::open(dir.path()).is_err());
    }
}

#[test]
fn mlx_adapter_separates_quantized_values_from_their_stored_words() {
    let source = MemorySource::new(
        json!({
            "w.weight": {"dtype": "U32", "shape": [2,8], "data_offsets": [0,64]},
            "w.scales": {"dtype": "BF16", "shape": [2,1], "data_offsets": [64,68]},
            "w.biases": {"dtype": "BF16", "shape": [2,1], "data_offsets": [68,72]}
        }),
        &[0; 72],
    );
    let inventory = read_safetensors(&source, ObjectId(0)).unwrap();
    let tensors = mlx::tensors(
        &inventory,
        &json!({"quantization":{"bits":4,"group_size":64}}),
    )
    .unwrap();

    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].shape(), Some([2, 64].as_slice()));
    assert_eq!(tensors[0].logical.as_ref().unwrap().dtype, None);
    assert_eq!(tensors[0].encoding.data().len(), 3);
    assert_eq!(inventory.tensors.len(), 3);

    let mut invalid = tensors[0].clone();

    if let TensorEncoding::Affine { group_axis, .. } = &mut invalid.encoding {
        *group_axis = usize::MAX;
    }

    assert!(invalid.validate().is_err());

    let mut invalid = tensors[0].clone();
    invalid.logical.as_mut().unwrap().shape[1] = 32;

    assert!(invalid.validate().is_err());
}
