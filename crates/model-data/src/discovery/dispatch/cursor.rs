//! Cursor envelopes bind provider positions to a store and an unchanged request.

use super::StoreId;
use crate::discovery::{Filter, SearchQuery};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

/// Request fields that must stay fixed while following a continuation.
/// Only serialized keys are compared; decoding numeric operands can round floats.
#[derive(Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub(super) enum RequestScope {
    /// Search predicates and page size, excluding the changing cursor.
    Search {
        text: Option<String>,
        filters: Vec<Filter>,
        limit: usize,
    },
    /// Inventory traversal has no search predicates.
    Enumerate { limit: usize },
}

/// Transport envelope for a provider-native cursor, not an authentication token.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cursor {
    store: StoreId,
    /// Preserve the serialized request exactly instead of reparsing its numbers.
    request_key: String,
    position: String,
}

impl RequestScope {
    /// Capture the search identity independently of its current page position.
    pub(super) fn search(query: &SearchQuery) -> Self {
        Self::Search {
            text: query.text.clone(),
            filters: query.filters.clone(),
            limit: query.page.limit,
        }
    }

    /// Return the native position after checking the store and serialized request.
    pub(super) fn unwrap_cursor(
        &self,
        store: &StoreId,
        cursor: Option<&str>,
    ) -> Result<Option<String>> {
        let Some(cursor) = cursor else {
            return Ok(None);
        };
        let cursor: Cursor = serde_json::from_str(cursor).context("invalid discovery cursor")?;

        ensure!(
            cursor.store == *store,
            "discovery cursor belongs to another store"
        );
        ensure!(
            cursor.request_key == self.key()?,
            "discovery cursor belongs to another request"
        );

        Ok(Some(cursor.position))
    }

    /// Bind a continuation to this request; preserve `None` at the end of a walk.
    pub(super) fn wrap_cursor(
        &self,
        store: &StoreId,
        position: Option<String>,
    ) -> Result<Option<String>> {
        position
            .map(|position| {
                serde_json::to_string(&Cursor {
                    store: store.clone(),
                    request_key: self.key()?,
                    position,
                })
                .context("encoding discovery cursor")
            })
            .transpose()
    }

    /// Encode a deterministic comparison key without a numeric decode round trip.
    /// Filter order remains significant, matching the unchanged-request contract.
    fn key(&self) -> Result<String> {
        serde_json::to_string(self).context("encoding discovery request key")
    }
}
