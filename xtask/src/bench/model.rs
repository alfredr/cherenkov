//! Resolve benchmark inputs through the CLI without linking the Metal engine.

use crate::util;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    process::Command,
};

/// Pinned index entry and its currently selected prepared artifact.
pub(super) struct Model {
    pub reference: String,
    pub directory: PathBuf,
    pub precisions: Vec<u8>,
    /// Original local checkpoint, used to recognize reports made before the index.
    pub source_directory: Option<PathBuf>,
}

impl Model {
    /// Use the same resolver as inference; preparation and inspection are untimed.
    pub(super) fn inspect(binary: &Path, selector: &OsStr, root: Option<&Path>) -> Result<Self> {
        let mut command = Command::new(binary);

        command.arg("inspect").arg(selector).arg("--json");

        if let Some(root) = root {
            command.arg("--root").arg(root);
        }

        let output = command.output().context("inspecting benchmark model")?;

        ensure!(
            output.status.success(),
            "model inspection failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );

        Self::from_inspection(
            &serde_json::from_slice(&output.stdout).context("decoding model inspection")?,
        )
    }

    /// Extract the index identity and prepared location from the public inspection view.
    fn from_inspection(value: &Value) -> Result<Self> {
        let index = &value["index"];
        let reference = index["id"]
            .as_str()
            .context("inspection missing model ID")?
            .to_owned();
        let artifact = index["artifacts"]
            .as_array()
            .context("inspection missing artifacts")?
            .iter()
            .find(|artifact| artifact["kind"] == "prepared" && artifact["available"] == true)
            .context(
                "model must be prepared before benchmarking; run `cherenkov prepare SOURCE`",
            )?;
        let directory = artifact["path"]
            .as_str()
            .context("prepared artifact missing path")?
            .into();
        let precisions = serde_json::from_value(index["precisions"].clone())
            .context("inspection missing expert precisions")?;

        Ok(Self {
            reference,
            directory,
            precisions,
            source_directory: index["source"]["path"].as_str().map(PathBuf::from),
        })
    }

    /// Upgrade a proven source identity without exposing its path in shared reports.
    pub(super) fn migrate_signature(&self, signature: &mut Value) {
        if signature["model_id"].is_string() {
            return;
        }

        let matches_id = signature["model"].as_str() == Some(&self.reference);
        let matches_path = self.source_directory.as_ref().is_some_and(|source| {
            let digest: String = Sha256::digest(source.as_os_str().as_encoded_bytes())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();

            signature["model"]
                .as_str()
                .is_some_and(|saved| Some(saved) == source.to_str())
                || signature["model_path_sha256"].as_str() == Some(&digest)
        });

        if !matches_id && !matches_path {
            return;
        }

        signature["model"] = json!("<model>");
        signature["model_id"] = json!(self.reference);

        signature
            .as_object_mut()
            .unwrap()
            .remove("model_path_sha256");

        if let Some(metadata) = signature["model_metadata_sha256"].as_object_mut()
            && let Some(manifest) = metadata.remove("packed/manifest.json")
        {
            metadata.entry("manifest.json").or_insert(manifest);
        }
    }

    /// Hash equivalent metadata in flat artifacts and legacy model/packed layouts.
    pub(super) fn metadata(&self) -> Result<Value> {
        let mut metadata = json!({});

        for name in ["config.json", "tokenizer.json"] {
            let direct = self.directory.join(name);
            let path = if direct.is_file() {
                direct
            } else {
                self.directory
                    .parent()
                    .context("model metadata missing")?
                    .join(name)
            };
            metadata[name] = json!(util::digest(&path)?);
        }

        metadata["manifest.json"] = json!(util::digest(&self.directory.join("manifest.json"))?);

        Ok(metadata)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/bench_model.rs"]
mod tests;
