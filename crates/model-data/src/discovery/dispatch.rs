//! Checked adapter calls, store-qualified results, and scoped continuations.

use super::{
    DiscoveryCapabilities, ModelCandidate, PageRequest, SearchPage, SearchQuery, StoreDiscovery,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

mod cursor;
use cursor::RequestScope;

/// Stable identity assigned by the store registry, independent of its display name.
/// IDs must be unique and persist across restarts and enable/disable operations.
/// A replacement store with a different source namespace needs a new ID so saved
/// locators and cursors cannot silently resolve against another source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StoreId(pub String);

/// Candidate with the store needed to resolve its provider-defined reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredCandidate {
    /// Registry identity of the originating store.
    pub store: StoreId,
    /// Metadata and locator supplied by that store.
    #[serde(flatten)]
    pub candidate: ModelCandidate,
}

/// Search request checked by [`Discovery`], with a provider-native cursor.
/// Validation covers the advertised schema, not the native cursor's contents or
/// whether returned candidates satisfy the filters. The adapter checks those.
pub struct ValidatedSearch<'a>(&'a SearchQuery);

impl ValidatedSearch<'_> {
    /// Borrow the checked query. Adapters still implement predicate semantics.
    pub fn query(&self) -> &SearchQuery {
        self.0
    }
}

/// Enumeration request checked by [`Discovery`], with a provider-native cursor.
/// The adapter must still reject invalid or expired native positions.
pub struct ValidatedPage<'a>(&'a PageRequest);

impl ValidatedPage<'_> {
    /// Borrow the checked page request.
    pub fn page(&self) -> &PageRequest {
        self.0
    }
}

/// Validates discovery requests and attaches store identity to results.
/// Constructing a dispatcher performs no I/O and does not enumerate the store.
/// Search and enumeration are synchronous and may block on the adapter's I/O.
/// Cursor envelopes detect accidental reuse; they do not authenticate input.
pub struct Discovery<'a> {
    store: StoreId,
    adapter: &'a dyn StoreDiscovery,
}

impl<'a> Discovery<'a> {
    /// Bind an adapter to its stable registry identity.
    /// The caller supplies a unique, persistent ID; this does not register a store.
    pub fn new(store: StoreId, adapter: &'a dyn StoreDiscovery) -> Self {
        Self { store, adapter }
    }

    /// Describe the adapter's search fields and enumeration limits.
    pub fn capabilities(&self) -> DiscoveryCapabilities {
        self.adapter.capabilities()
    }

    /// Validate and search one page, preserving the source of every result.
    pub fn search(&self, query: &SearchQuery) -> Result<SearchPage<StoredCandidate>> {
        let capabilities = self.capabilities();
        let schema = capabilities
            .search
            .context("this store does not support search")?;

        schema.validate(query)?;

        let scope = RequestScope::search(query);
        let mut native = query.clone();

        native.page.cursor = scope.unwrap_cursor(&self.store, query.page.cursor.as_deref())?;

        let page = self.adapter.search(ValidatedSearch(&native))?;

        self.finish(page, &scope, &native.page)
    }

    /// Validate and enumerate one page; search support alone does not allow this.
    pub fn enumerate(&self, request: &PageRequest) -> Result<SearchPage<StoredCandidate>> {
        let maximum = self
            .capabilities()
            .enumeration_max_page_size
            .context("this store does not support enumeration")?;

        request.validate(maximum)?;

        let scope = RequestScope::Enumerate {
            limit: request.limit,
        };
        let mut native = request.clone();

        native.cursor = scope.unwrap_cursor(&self.store, request.cursor.as_deref())?;

        let page = self.adapter.enumerate(ValidatedPage(&native))?;

        self.finish(page, &scope, &native)
    }

    /// Enforce response bounds, reject a stalled cursor, and preserve result origin.
    /// Empty result pages are valid when filtering leaves a continuation to follow.
    fn finish(
        &self,
        page: SearchPage,
        scope: &RequestScope,
        request: &PageRequest,
    ) -> Result<SearchPage<StoredCandidate>> {
        ensure!(
            page.items.len() <= request.limit,
            "store exceeded the requested page limit"
        );
        ensure!(
            page.next_cursor.is_none() || page.next_cursor != request.cursor,
            "store returned a continuation cursor that did not advance"
        );

        Ok(SearchPage {
            items: page
                .items
                .into_iter()
                .map(|candidate| StoredCandidate {
                    store: self.store.clone(),
                    candidate,
                })
                .collect(),
            next_cursor: scope.wrap_cursor(&self.store, page.next_cursor)?,
            total: page.total,
            gaps: page.gaps,
        })
    }
}
