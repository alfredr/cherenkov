//! Catalog lookup accepts aliases and source URIs without dropping source identity.

use super::{
    Catalog, ModelEntry, Source, disk,
    selector::{self, Selector},
};
use anyhow::{Context, Result, ensure};
use std::path::Path;

/// Resolve exactly one catalog entry; never fall back from an ambiguous source.
pub(super) fn lookup<'a>(catalog: &'a Catalog, reference: &str) -> Result<&'a ModelEntry> {
    let selector = selector::parse(Path::new(reference))?;

    find(catalog, &selector, reference)?
        .with_context(|| format!("model {reference:?} is not registered"))
}

/// Lookup may report absence so the resolver can register explicit sources.
/// Store-qualified references inspect catalog metadata, not live store contents.
pub(super) fn find<'a>(
    catalog: &'a Catalog,
    selector: &Selector,
    reference: &str,
) -> Result<Option<&'a ModelEntry>> {
    if let Selector::Registered(key) = selector {
        return Ok(catalog.models.get(key).or_else(|| {
            catalog
                .models
                .values()
                .find(|model| model.name.as_deref() == Some(key.as_str()))
        }));
    }

    if let Selector::Disk { store, .. } = selector {
        disk::registered(catalog, store)?;
    }

    let mut matches = catalog
        .models
        .values()
        .filter(|model| matches_selector(catalog, &model.source, selector));
    let first = matches.next();

    ensure!(
        matches.next().is_none(),
        "model reference {reference:?} is ambiguous; append @revision or use an alias or model ID"
    );

    Ok(first)
}

/// Prefer an alias or unqualified source URI; add revision detail only when needed.
pub(super) fn preferred(catalog: &Catalog, model: &ModelEntry) -> String {
    if let Some(name) = &model.name {
        return name.clone();
    }

    let (reference, revision) = display_source(catalog, &model.source);

    if selects(catalog, &reference, &model.id) {
        return reference;
    }

    if let Some(revision) = revision
        && is_revision(&revision)
    {
        for length in 8..=revision.len() {
            let qualified = format!("{reference}@{}", &revision[..length]);

            if selects(catalog, &qualified, &model.id) {
                return qualified;
            }
        }
    }

    model.id.clone()
}

/// Choose a concise source locator, keeping revision detail for disambiguation.
fn display_source(catalog: &Catalog, source: &Source) -> (String, Option<String>) {
    match source {
        Source::HuggingFace { repo, revision, .. } => {
            (format!("hf://{repo}"), Some(revision.clone()))
        }
        Source::Local { path, fingerprint } => {
            for store in catalog.stores.values().filter(|store| store.enabled) {
                if let Some(Selector::Disk {
                    store,
                    repo,
                    revision,
                }) = store.selector(path)
                {
                    return (
                        format!("disk://{store}/{repo}"),
                        Some(revision.unwrap_or_else(|| fingerprint.clone())),
                    );
                }
            }

            (path.to_string_lossy().into_owned(), None)
        }
    }
}

/// Confirm that a display selector returns this entry, including alias precedence.
fn selects(catalog: &Catalog, reference: &str, id: &str) -> bool {
    lookup(catalog, reference).is_ok_and(|model| model.id == id)
}

/// Match an exact source namespace and optional immutable revision prefix.
fn matches_selector(catalog: &Catalog, source: &Source, selector: &Selector) -> bool {
    match (source, selector) {
        (
            Source::HuggingFace { repo, revision, .. },
            Selector::Hub {
                repo: wanted,
                revision: rev,
            },
        ) => {
            repo == wanted
                && rev
                    .as_deref()
                    .is_none_or(|r| is_revision(r) && revision.starts_with(r))
        }
        (Source::Local { path, .. }, Selector::Path(wanted)) => {
            let absolute = crate::config::absolute(wanted).ok();

            absolute.is_some_and(|p| path == &p.canonicalize().unwrap_or(p))
        }
        (
            Source::Local { path, fingerprint },
            Selector::Disk {
                store,
                repo,
                revision,
            },
        ) => {
            let Some(store) = catalog.stores.get(store) else {
                return false;
            };
            let Some(Selector::Disk {
                repo: found,
                revision: commit,
                ..
            }) = store.selector(path)
            else {
                return false;
            };

            found == *repo
                && revision.as_deref().is_none_or(|r| {
                    is_revision(r) && commit.as_deref().unwrap_or(fingerprint).starts_with(r)
                })
        }
        (
            Source::Local { path, fingerprint },
            Selector::LocalRevision {
                path: wanted,
                revision,
            },
        ) => path == wanted && fingerprint.starts_with(revision),
        _ => false,
    }
}

/// Immutable revision prefixes use at least eight hexadecimal digits.
pub(super) fn is_revision(value: &str) -> bool {
    value.len() >= 8 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
