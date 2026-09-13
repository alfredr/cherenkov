//! Pinned Hugging Face downloads. The Hub client handles retries, locking and Xet transfers.

use crate::units::{BYTES_PER_GB, BYTES_PER_MIB};
use crate::{
    qwen4_exp::Qwen4ExpConfig,
    storage::{self, Paths},
};
use anyhow::{Context, Result, ensure};
use hf_hub::{HFClient, HFClientSync, repository::RepoTreeEntry};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

const METADATA: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "NOTICE",
    "NOTICE.txt",
    "NOTICE.md",
];

pub struct Download<'a> {
    pub repo: &'a str,
    pub revision: &'a str,
    pub token: Option<&'a str>,
    pub metadata_only: bool,
}

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

pub fn run(paths: &Paths, request: Download<'_>) -> Result<PathBuf> {
    run_at(paths, request, None)
}

pub(crate) fn run_at(
    paths: &Paths,
    request: Download<'_>,
    endpoint: Option<&str>,
) -> Result<PathBuf> {
    // Validate path components before contacting the Hub. Branch names themselves
    // are sent to the API, then replaced with the returned immutable commit.
    paths.model(request.repo, storage::DEFAULT_REVISION)?;
    storage::create_private_dir(&paths.data)?;
    storage::create_private_dir(&paths.scratch)?;

    let lock = std::fs::File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(paths.data.join("download.lock"))?;

    lock.try_lock()
        .context("another download is using this root")?;

    let mut builder = HFClient::builder().cache_dir(paths.downloads());

    if let Some(endpoint) = endpoint {
        builder = builder.endpoint(endpoint);
    }

    if let Some(token) = request.token {
        ensure!(!token.trim().is_empty(), "HF token must not be empty");

        builder = builder.token(token);
    }

    let client = HFClientSync::from_inner(builder.build()?)?;
    let (owner, name) = hf_hub::split_id(request.repo);
    let repo = client.model(owner, name);
    let info = repo.info().revision(request.revision).send()?;
    let commit = info.sha.context("Hub response has no commit hash")?;
    let model = paths.model(request.repo, &commit)?;
    let snapshot = paths.snapshot(request.repo, &commit)?;
    let entries = repo.list_tree().revision(&commit).recursive(false).send()?;
    let available: BTreeMap<String, u64> = entries
        .into_iter()
        .filter_map(|e| match e {
            RepoTreeEntry::File { path, size, .. } => Some((path, size)),
            _ => None,
        })
        .collect();

    ensure!(
        available.contains_key("config.json") && available.contains_key("tokenizer.json"),
        "checkpoint needs config.json and tokenizer.json"
    );

    let mut files: BTreeSet<String> = METADATA
        .iter()
        .filter(|f| available.contains_key(**f))
        .map(|s| (*s).into())
        .collect();

    eprintln!(
        "downloading {} at {commit} into {}",
        request.repo,
        model.display()
    );

    for name in &files {
        ensure!(
            available[name] <= 64 * BYTES_PER_MIB as u64,
            "metadata file {name} exceeds 64 MiB"
        );
    }

    storage::require_space(&paths.data, missing_bytes(&files, &available, &snapshot)?)?;
    // Validate the architecture before fetching tokenizer or weight payloads.
    repo.download_file()
        .filename("config.json")
        .revision(&commit)
        .send()?;
    Qwen4ExpConfig::load(&snapshot)?;

    if !request.metadata_only {
        if available.contains_key("model.safetensors.index.json") {
            let index = repo
                .download_file()
                .filename("model.safetensors.index.json")
                .revision(&commit)
                .send()?;

            files.extend(shards(&std::fs::read(index)?)?);
        } else {
            files.insert("model.safetensors".into());
        }
    }

    let needed = missing_bytes(&files, &available, &snapshot)?;

    storage::require_space(&paths.data, needed)?;
    storage::require_space(&paths.scratch, 0)?;
    eprintln!(
        "{} selected files, {:.2} GB not yet cached",
        files.len(),
        needed as f64 / BYTES_PER_GB as f64
    );

    let snapshot = repo
        .snapshot_download()
        .revision(&commit)
        .allow_patterns(files.iter().cloned().collect())
        .max_workers(4)
        .progress(progress::Reporter::default())
        .send()?;

    // Publish only completed, size-checked downloads. Links share durable Hub
    // blobs; packed and low-bit stores live separately in the model's packed/.
    ensure!(
        missing_bytes(&files, &available, &snapshot)? == 0,
        "download did not produce all selected files at their expected sizes"
    );
    publish(&model, &snapshot, &files)?;

    Ok(model)
}

fn shards(bytes: &[u8]) -> Result<BTreeSet<String>> {
    let index: Index = serde_json::from_slice(bytes).context("safetensors index")?;
    let shards: BTreeSet<_> = index.weight_map.into_values().collect();

    ensure!(!shards.is_empty(), "empty safetensors index");

    for shard in &shards {
        ensure!(
            storage::safe_component(shard) && shard.ends_with(".safetensors"),
            "unsupported shard filename {shard:?}"
        );
    }

    Ok(shards)
}

fn missing_bytes(
    files: &BTreeSet<String>,
    available: &BTreeMap<String, u64>,
    snapshot: &Path,
) -> Result<u64> {
    let mut total = 0u64;

    for file in files {
        ensure!(
            storage::safe_component(file),
            "unsupported filename {file:?}"
        );

        let size = *available
            .get(file)
            .with_context(|| format!("checkpoint is missing {file}"))?;

        if !std::fs::metadata(snapshot.join(file)).is_ok_and(|m| m.is_file() && m.len() == size) {
            total = total.checked_add(size).context("download size overflow")?;
        }
    }

    Ok(total)
}

fn publish(model: &Path, snapshot: &Path, files: &BTreeSet<String>) -> Result<()> {
    storage::create_private_dir(model)?;

    for file in files {
        let dest = model.join(file);
        let source = snapshot.join(file).canonicalize()?;

        match std::fs::symlink_metadata(&dest) {
            Ok(_) => ensure!(
                dest.canonicalize()? == source,
                "refusing to replace {}",
                dest.display()
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                std::os::unix::fs::symlink(source, dest)?
            }
            Err(e) => return Err(e.into()),
        }
    }

    Ok(())
}

mod progress;
#[cfg(test)]
#[path = "../tests/unit/download.rs"]
mod tests;
