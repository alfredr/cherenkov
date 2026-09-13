//! Parse locator syntax once, then map its structured fields through the index.

use anyhow::{Context, Result, ensure};
use reqwest::Url;
use std::path::{Path, PathBuf};

/// Parsed input; each variant selects an index lookup or a source adapter.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Selector {
    Registered(String),
    Hub {
        repo: String,
        revision: Option<String>,
    },
    Disk {
        store: String,
        repo: String,
        revision: Option<String>,
    },
    Path(PathBuf),
    LocalRevision {
        path: PathBuf,
        revision: String,
    },
}

/// Explicit paths take precedence; all URI forms go through the URL parser.
pub(super) fn parse(input: &Path) -> Result<Selector> {
    let Some(text) = input.to_str() else {
        return Ok(Selector::Path(input.to_owned()));
    };

    ensure!(!text.is_empty(), "empty model reference");

    if input.is_absolute()
        || matches!(text, "." | "..")
        || text.starts_with("./")
        || text.starts_with("../")
    {
        return Ok(Selector::Path(input.to_owned()));
    }

    let mut components = input.components();

    if components
        .next()
        .is_some_and(|part| part.as_os_str() == "~")
    {
        return Ok(Selector::Path(
            dirs::home_dir()
                .context("home directory unavailable")?
                .join(components.as_path()),
        ));
    }

    let uri = match Url::parse(text) {
        Ok(uri) => uri,
        Err(_) if !text.contains(':') => {
            return Ok(if text.contains('/') {
                Selector::Path(input.to_owned())
            } else {
                Selector::Registered(text.to_owned())
            });
        }
        Err(error) => return Err(error).context("invalid model locator"),
    };

    ensure!(
        uri.query().is_none() && uri.fragment().is_none(),
        "model locators do not accept queries or fragments"
    );

    match uri.scheme() {
        "model" => {
            ensure!(
                uri.cannot_be_a_base(),
                "use model:alias for legacy references"
            );

            legacy(uri.path())
        }
        "hf" | "disk" => source(&uri, text),
        _ => anyhow::bail!("unsupported model source scheme {:?}", uri.scheme()),
    }
}

/// Map a parsed URI authority and repository path into a source selector.
fn source(uri: &Url, original: &str) -> Result<Selector> {
    ensure!(
        uri.username().is_empty() && uri.password().is_none() && uri.port().is_none(),
        "model source authority cannot contain credentials or a port"
    );
    ensure!(
        !original.contains('%')
            && !uri.as_str().contains('%')
            && !original.split('/').any(|part| matches!(part, "." | "..")),
        "model source paths must use literal names without dot segments"
    );

    let authority = uri
        .host_str()
        .context("model source requires an authority")?;

    ensure!(
        crate::storage::safe_component(authority),
        "invalid source authority"
    );

    let segments: Vec<_> = uri
        .path_segments()
        .context("model source requires a repository path")?
        .collect();
    let location = segments.join("/");
    let (repository, revision) = revision(&location)?;

    match uri.scheme() {
        "hf" => {
            ensure!(
                crate::storage::safe_component(repository),
                "use hf://owner/repo[@revision]"
            );

            Ok(Selector::Hub {
                repo: format!("{authority}/{repository}"),
                revision,
            })
        }
        "disk" => {
            let (owner, name) = repository
                .split_once('/')
                .context("use disk://store/owner/repo[@revision]")?;

            ensure!(
                crate::storage::safe_component(owner) && crate::storage::safe_component(name),
                "use disk://store/owner/repo[@revision]"
            );

            Ok(Selector::Disk {
                store: authority.to_owned(),
                repo: repository.to_owned(),
                revision,
            })
        }
        _ => unreachable!(),
    }
}

/// Separate the optional revision delimiter from the repository path.
fn revision(location: &str) -> Result<(&str, Option<String>)> {
    let Some((repository, revision)) = location.split_once('@') else {
        return Ok((location, None));
    };

    validate_revision(revision)?;

    Ok((repository, Some(revision.to_owned())))
}

/// Apply the same literal revision rules to URI suffixes and option overrides.
pub(super) fn validate_revision(revision: &str) -> Result<()> {
    ensure!(
        revision.split('/').all(crate::storage::safe_component),
        "invalid model revision"
    );

    Ok(())
}

/// Map older opaque model: locators onto the same structured selectors.
fn legacy(key: &str) -> Result<Selector> {
    ensure!(!key.is_empty(), "empty model reference");

    let Some((kind, location)) = key.split_once(':') else {
        return Ok(Selector::Registered(key.to_owned()));
    };
    let (location, revision) = location
        .rsplit_once(':')
        .context("legacy source reference needs a revision")?;

    ensure!(
        super::reference::is_revision(revision),
        "invalid legacy revision"
    );

    match kind {
        "hf" => {
            let (owner, repo) = location
                .split_once('/')
                .context("HF source requires owner/repo")?;

            ensure!(
                crate::storage::safe_component(owner) && crate::storage::safe_component(repo),
                "invalid HF repository"
            );

            Ok(Selector::Hub {
                repo: location.to_owned(),
                revision: Some(revision.to_owned()),
            })
        }
        "local" => {
            ensure!(
                Path::new(location).is_absolute(),
                "legacy local reference requires an absolute path"
            );

            Ok(Selector::LocalRevision {
                path: PathBuf::from(location),
                revision: revision.to_owned(),
            })
        }
        _ => anyhow::bail!("unknown legacy source kind"),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/model/index/selector.rs"]
mod tests;
