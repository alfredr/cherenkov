//! XDG locations and the durable model layout. Resolving paths never creates files.

use crate::units::BYTES_PER_GB;
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// HF repository used when no model is selected.
pub const DEFAULT_REPO: &str = "Sawfwair/Qwen3.8-Flash-Next-MLX-4bit";
/// Immutable revision of the tested default checkpoint.
pub const DEFAULT_REVISION: &str = "6cc9bbc0fae9ce26b7670b3ed1e26d557c154506";

/// Source selector shared by preparation and serving when no model is specified.
pub fn default_model_reference() -> String {
    format!("hf://{DEFAULT_REPO}@{DEFAULT_REVISION}")
}

/// Resolved application locations, using XDG directories on macOS and Linux.
#[derive(Debug, Clone, Serialize)]
pub struct Paths {
    /// Downloaded checkpoints and generated stores; never automatically evicted.
    pub data: PathBuf,
    /// Disposable transfer scratch. Inference checkpoints remain in RAM.
    pub scratch: PathBuf,
    /// Server configuration file.
    pub config: PathBuf,
}

impl Paths {
    /// Resolve locations from an explicit root or absolute XDG overrides.
    /// An explicit root holds data, `scratch/`, and `cherenkov.toml` together.
    /// Without overrides, use the user's `.local/share`, `.cache`, and `.config`
    /// directories. This does not create directories or read configuration files.
    pub fn new(root: Option<&Path>) -> Result<Self> {
        if let Some(root) = root {
            let root = crate::config::absolute(root)?;

            return Ok(Self {
                scratch: root.join("scratch"),
                config: root.join("cherenkov.toml"),
                data: root,
            });
        }

        // Use the same dot-directory layout on macOS and Linux.
        let home = dirs::home_dir();
        let base =
            |variable, fallback| xdg_dir(std::env::var_os(variable), home.as_deref(), fallback);

        Ok(Self {
            data: base("XDG_DATA_HOME", ".local/share")?.join("cherenkov"),
            scratch: base("XDG_CACHE_HOME", ".cache")?.join("cherenkov"),
            config: base("XDG_CONFIG_HOME", ".config")?.join("cherenkov/cherenkov.toml"),
        })
    }

    /// HF download cache beneath the data directory.
    pub fn downloads(&self) -> PathBuf {
        self.data.join("downloads")
    }

    /// Resolve a model directory from `owner/name` and a full commit hash.
    /// Reject unsafe path components and revisions that are not 40 hexadecimal digits.
    pub fn model(&self, repo: &str, commit: &str) -> Result<PathBuf> {
        validate_identity(repo, commit)?;

        Ok(self.data.join("models").join(repo).join(commit))
    }

    /// Resolve the built-in checkpoint's directory without checking availability.
    pub fn default_model(&self) -> PathBuf {
        self.model(DEFAULT_REPO, DEFAULT_REVISION)
            .expect("built-in model identity")
    }

    /// The Hub's documented snapshot layout, also used to count already cached bytes.
    pub(crate) fn snapshot(&self, repo: &str, commit: &str) -> Result<PathBuf> {
        validate_identity(repo, commit)?;

        Ok(self
            .downloads()
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(commit))
    }
}

fn xdg_dir(value: Option<OsString>, home: Option<&Path>, fallback: &str) -> Result<PathBuf> {
    // XDG overrides must be absolute; empty and relative values use the default.
    if let Some(path) = value.map(PathBuf::from).filter(|path| path.is_absolute()) {
        return Ok(path);
    }

    let home =
        home.context("home directory unavailable; supply --root or absolute XDG directories")?;

    Ok(home.join(fallback))
}

fn validate_identity(repo: &str, commit: &str) -> Result<()> {
    let parts: Vec<_> = repo.split('/').collect();

    ensure!(
        parts.len() == 2 && parts.iter().all(|p| safe_component(p)),
        "model ID must be owner/name"
    );
    ensure!(
        commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit()),
        "model revision must resolve to a full commit hash"
    );

    Ok(())
}

pub(crate) fn safe_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
}

pub(crate) fn create_private_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;

    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(path)
        .with_context(|| format!("creating {}", path.display()))
}

/// Refuse before large writes; reserve space for filesystem and transfer overhead.
pub(crate) fn require_space(path: &Path, additional_bytes: u64) -> Result<()> {
    let path_c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())?;
    let mut stat: libc::statfs = unsafe { std::mem::zeroed() };

    if unsafe { libc::statfs(path_c.as_ptr(), &mut stat) } != 0 {
        return Err(std::io::Error::last_os_error()).context("checking available disk space");
    }

    let free = stat.f_bavail as u64 * stat.f_bsize as u64;
    let required = additional_bytes
        .checked_add(2_000_000_000)
        .context("disk budget overflow")?;

    ensure!(
        free >= required,
        "{} needs {:.1} GB more disk space plus a 2 GB reserve; only {:.1} GB is available",
        path.display(),
        additional_bytes as f64 / BYTES_PER_GB as f64,
        free as f64 / BYTES_PER_GB as f64
    );

    Ok(())
}

#[cfg(test)]
pub(crate) fn test_model_dir() -> Option<PathBuf> {
    let path = std::env::var_os("CHERENKOV_MODEL_DIR")
        .map(PathBuf::from)
        .or_else(|| Paths::new(None).ok().map(|p| p.default_model()))?;

    path.join("packed/manifest.json").exists().then_some(path)
}

#[cfg(test)]
#[path = "../tests/unit/storage.rs"]
mod tests;
