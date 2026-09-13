use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

/// Metadata identity keeps registration cheap. Packing checks it again before
/// publication. Files must not be modified while their tensor mappings are live.
pub(super) fn fingerprint(path: &Path) -> Result<String> {
    let files = if path.is_file() {
        vec![path.to_owned()]
    } else {
        let mut files = std::fs::read_dir(path)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;

        files.retain(|p| {
            p.is_file()
                && p.extension().is_some_and(|e| {
                    matches!(
                        e.to_str(),
                        Some("json" | "jinja" | "safetensors" | "gguf" | "bin")
                    )
                })
        });
        files.sort();

        files
    };
    let mut hash = Sha256::new();

    for file in files {
        let meta = std::fs::metadata(&file)?;

        hash.update(
            file.file_name()
                .context("missing filename")?
                .as_encoded_bytes(),
        );
        hash.update(meta.len().to_le_bytes());
        hash.update(
            meta.modified()?
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
                .to_le_bytes(),
        );

        if file
            .extension()
            .is_some_and(|e| e == "json" || e == "jinja")
        {
            ensure!(
                meta.len() <= 64 * 1024 * 1024,
                "metadata file exceeds 64 MiB"
            );

            let mut reader = std::fs::File::open(file)?;
            let mut buf = [0_u8; 65536];

            loop {
                let len = reader.read(&mut buf)?;

                if len == 0 {
                    break;
                }

                hash.update(&buf[..len]);
            }
        }
    }

    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub(super) fn copy_store(source: &Path, target: &Path, link: bool) -> Result<()> {
    // Published managed inputs are pinned by a lease. Hard links reuse their
    // immutable base files; newly selected variants are written under target.
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;

        if !entry.path().is_file() {
            continue;
        }

        let dest = target.join(entry.file_name());

        if link && entry.path().extension().is_some_and(|e| e == "bin") {
            std::fs::hard_link(entry.path(), &dest)
                .or_else(|_| std::fs::copy(entry.path(), &dest).map(|_| ()))?;
        } else {
            std::fs::copy(entry.path(), dest)?;
        }
    }

    Ok(())
}
