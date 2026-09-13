use super::*;

pub(super) fn open(dir: &Path) -> Result<Checkpoint> {
    let manifest = Manifest::load(dir)?;
    let config_path = if dir.join("config.json").is_file() {
        dir.join("config.json")
    } else {
        dir.parent()
            .context("packed configuration missing")?
            .join("config.json")
    };
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(config_path)?)?;
    let paths = ["dense.bin", "experts.bin", "ngram.bin"].map(|name| dir.join(name));
    let objects = MappedObjects::open(paths.iter().map(|path| path.as_path()))?;
    let mut tensors = dense(&manifest)?;

    experts(&manifest, &mut tensors)?;

    let table_id = TensorId(tensors.len());
    let table = ngram(&manifest)?;

    tensors.push(table);

    let text = config.get("text_config").unwrap_or(&config);
    let n = &manifest.ngram;
    let ngram = Some(NgramTable {
        hashing: NgramHash::Qwen4Exp {
            ngram_size: text["ngram_size"].as_u64().unwrap_or(3),
            heads_per_ngram: text["heads_per_ngram"].as_u64().unwrap_or(8),
            head_offsets: n.head_offsets.clone(),
            head_vocab_sizes: n.head_vocab_sizes.clone(),
            layer_multipliers: n.layer_multipliers.clone(),
        },
        shards: vec![TableShard {
            first_row: 0,
            rows: n.rows,
            tensor: table_id,
        }],
    });
    let description = ModelDescription {
        schema_version: 1,
        format: CheckpointFormat {
            container: ContainerFormat::CherenkovPacked {
                version: manifest.version,
            },
            conventions: CheckpointConventions::Qwen4ExpCherenkov,
        },
        architecture: architecture(&config),
        metadata: serde_json::json!({"config": config, "manifest": manifest}),
        tensors,
        ngram,
    };
    let checkpoint = Checkpoint {
        description,
        objects,
    };

    for tensor in &checkpoint.description.tensors {
        tensor.validate()?;

        for data in tensor.encoding.data() {
            checkpoint.map(data)?;
        }
    }

    Ok(checkpoint)
}

fn dense(manifest: &Manifest) -> Result<Vec<Tensor>> {
    let mut tensors = Vec::new();

    for entry in &manifest.dense {
        if entry.name.ends_with(".scales") || entry.name.ends_with(".biases") {
            continue;
        }

        let codes = dense_part(entry)?;
        let role = qwen_role(&entry.name, entry.shape.len());

        if codes.dtype != Dtype::U32 || !entry.name.ends_with(".weight") {
            tensors.push(Tensor::new(
                entry.name.clone(),
                role,
                Some(codes.shape.clone()),
                TensorEncoding::Dense { tensor: codes },
            ));

            continue;
        }

        let prefix = entry.name.trim_end_matches(".weight");
        let scales = dense_part(manifest.dense(&format!("{prefix}.scales"))?)?;
        let biases = dense_part(manifest.dense(&format!("{prefix}.biases"))?)?;

        tensors.push(affine(entry.name.clone(), role, codes, scales, biases)?);
    }

    Ok(tensors)
}

fn dense_part(entry: &crate::qwen4_exp::DenseEntry) -> Result<StoredTensor> {
    let dtype = match entry.dtype.as_str() {
        "U32" => Dtype::U32,
        "I64" => Dtype::I64,
        "F32" => Dtype::F32,
        "F16" => Dtype::F16,
        "BF16" => Dtype::Bf16,
        other => anyhow::bail!("unsupported packed dtype {other}"),
    };

    StoredTensor::contiguous(
        dtype,
        entry.shape.iter().map(|&v| v as u64).collect(),
        DataSpan {
            object: ObjectId(0),
            offset: entry.offset,
            length: entry.nbytes,
        },
    )
}

fn experts(manifest: &Manifest, tensors: &mut Vec<Tensor>) -> Result<()> {
    let l = &manifest.experts;

    for (layer, prefix) in l.layer_prefixes.iter().enumerate() {
        let base = (layer as u64)
            .checked_mul(l.experts as u64)
            .and_then(|v| v.checked_mul(l.record_stride))
            .context("expert offset overflow")?;
        let records = Records {
            object: ObjectId(1),
            base,
            count: l.experts as u64,
            stride: l.record_stride,
        };

        for (name, rows, width, offsets) in [
            (
                "gate_proj",
                l.inter,
                l.hidden,
                [l.gate_w, l.gate_s, l.gate_b],
            ),
            ("up_proj", l.inter, l.hidden, [l.up_w, l.up_s, l.up_b]),
            (
                "down_proj",
                l.hidden,
                l.inter,
                [l.down_w, l.down_s, l.down_b],
            ),
        ] {
            ensure!(
                l.group > 0 && width.is_multiple_of(l.group) && width.is_multiple_of(8),
                "invalid expert quantization group"
            );

            let codes = records.part(Dtype::U32, rows as u64, (width / 8) as u64, offsets[0])?;
            let scales = records.part(
                Dtype::Bf16,
                rows as u64,
                (width / l.group) as u64,
                offsets[1],
            )?;
            let biases = records.part(
                Dtype::Bf16,
                rows as u64,
                (width / l.group) as u64,
                offsets[2],
            )?;

            tensors.push(affine(
                format!("{prefix}.{name}.weight"),
                TensorRole::Expert,
                codes,
                scales,
                biases,
            )?);
        }
    }

    Ok(())
}

pub(super) fn ngram(manifest: &Manifest) -> Result<Tensor> {
    let n = &manifest.ngram;

    ensure!(
        n.group > 0 && n.dim.is_multiple_of(n.group) && n.dim.is_multiple_of(8),
        "invalid n-gram quantization group"
    );
    ensure!(
        n.weight_bytes == n.dim as u64 / 2 && n.scale_bytes == (n.dim / n.group * 2) as u64,
        "n-gram component sizes disagree with dimensions"
    );
    ensure!(
        n.row_bytes >= n.weight_bytes + 2 * n.scale_bytes,
        "n-gram record is too small"
    );

    let records = Records {
        object: ObjectId(2),
        base: 0,
        count: n.rows,
        stride: n.row_bytes,
    };
    let row_part = |dtype, width, offset| -> Result<StoredTensor> {
        let mut part = records.part(dtype, 1, width, offset)?;

        part.shape.remove(1);
        part.byte_strides.remove(1);

        Ok(part)
    };
    let codes = row_part(Dtype::U32, n.dim as u64 / 8, 0)?;
    let scales = row_part(Dtype::Bf16, (n.dim / n.group) as u64, n.weight_bytes)?;
    let biases = row_part(
        Dtype::Bf16,
        (n.dim / n.group) as u64,
        n.weight_bytes + n.scale_bytes,
    )?;

    affine(
        "ngram_embedding".into(),
        TensorRole::NgramEmbedding,
        codes,
        scales,
        biases,
    )
}

struct Records {
    object: ObjectId,
    base: u64,
    count: u64,
    stride: u64,
}

impl Records {
    fn part(&self, dtype: Dtype, rows: u64, width: u64, offset: u64) -> Result<StoredTensor> {
        ensure!(
            self.count > 0 && rows > 0 && width > 0,
            "empty packed record component"
        );

        let row_bytes = width
            .checked_mul(dtype.bytes())
            .context("row size overflow")?;
        let part_bytes = rows
            .checked_mul(row_bytes)
            .context("record size overflow")?;

        ensure!(
            offset
                .checked_add(part_bytes)
                .is_some_and(|end| end <= self.stride),
            "component exceeds record stride"
        );

        let length = (self.count - 1)
            .checked_mul(self.stride)
            .and_then(|v| v.checked_add(part_bytes))
            .context("record span overflow")?;
        let offset = self
            .base
            .checked_add(offset)
            .context("record offset overflow")?;
        let part = StoredTensor {
            dtype,
            shape: vec![self.count, rows, width],
            byte_strides: vec![self.stride, row_bytes, dtype.bytes()],
            data: DataSpan {
                object: self.object,
                offset,
                length,
            },
        };

        part.validate()?;

        Ok(part)
    }
}

fn affine(
    name: String,
    role: TensorRole,
    codes: StoredTensor,
    scales: StoredTensor,
    biases: StoredTensor,
) -> Result<Tensor> {
    let axis = codes
        .shape
        .len()
        .checked_sub(1)
        .context("scalar packed codes")?;
    let mut shape = codes.shape.clone();
    shape[axis] = shape[axis]
        .checked_mul(8)
        .context("packed width overflow")?;

    ensure!(
        scales.shape.len() == shape.len() && biases.shape == scales.shape,
        "affine component rank mismatch"
    );
    ensure!(
        scales.shape[..axis] == shape[..axis],
        "affine component row mismatch"
    );

    let groups = scales.shape[axis];

    ensure!(
        groups > 0 && shape[axis].is_multiple_of(groups),
        "invalid affine group count"
    );

    Ok(Tensor::new(
        name,
        role,
        Some(shape.clone()),
        TensorEncoding::Affine {
            bits: 4,
            group_size: shape[axis] / groups,
            group_axis: axis,
            packing: BitPacking::LowFirstU32,
            codes,
            scales,
            offset: AffineOffset::Bias { tensor: biases },
        },
    ))
}
