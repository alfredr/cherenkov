//! Store-defined model discovery, separate from reading tensor bytes.
//!
//! Providers advertise their filters. Common result metadata is optional and
//! does not determine execution compatibility or replace a checkpoint inventory.
//!
//! Adapters declare parameters through [`SearchSchema`] and implement
//! [`StoreDiscovery`]. Callers use [`Discovery`] to validate requests, attach store
//! identities, and follow continuation pages:
//!
//! ```
//! use cherenkov_model_data::discovery::*;
//! use serde_json::json;
//!
//! fn browse(adapter: &dyn StoreDiscovery, id: StoreId) -> anyhow::Result<()> {
//!     let store = Discovery::new(id, adapter);
//!     let mut query = SearchQuery {
//!         text: None,
//!         filters: vec![Filter {
//!             field: "author".to_owned(),
//!             operator: FilterOperator::Equal,
//!             value: json!("example-team"),
//!         }],
//!         page: PageRequest { limit: 20, cursor: None },
//!     };
//!
//!     loop {
//!         let page = store.search(&query)?;
//!         for item in page.items {
//!             println!("{}: {}", item.store.0, item.candidate.reference);
//!         }
//!         query.page.cursor = page.next_cursor;
//!         if query.page.cursor.is_none() {
//!             return Ok(());
//!         }
//!     }
//! }
//! ```
//!
//! An empty page can have a continuation. Keep the query and page size unchanged
//! and pass the cursor back verbatim. Adapters remain responsible for native
//! cursor validity and for returning candidates that satisfy every predicate.

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

mod dispatch;
mod query;
pub use dispatch::{Discovery, StoreId, StoredCandidate, ValidatedPage, ValidatedSearch};

/// Discovery operations supported by a store.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscoveryCapabilities {
    /// Maximum page size for walking the inventory, absent when unsupported.
    pub enumeration_max_page_size: Option<usize>,
    /// Search fields and limits, absent when search is unsupported.
    pub search: Option<SearchSchema>,
}

/// Optional discovery operations exposed by a file-store adapter.
/// Discovery returns candidates without registering models or fetching weights.
/// Methods are synchronous; search and enumeration may block on I/O.
/// Advertise only operations and filter semantics the adapter can honor.
pub trait StoreDiscovery {
    /// Describe supported operations and provider-specific filters without I/O.
    fn capabilities(&self) -> DiscoveryCapabilities;

    /// Search one page after [`Discovery`] validates the request.
    /// Implement text and filter semantics, combining predicates with AND.
    /// Unknown metadata must not be treated as a matching value, even for `NotEqual`.
    /// Report known omissions from missing filter metadata in [`SearchPage::gaps`].
    /// The supplied cursor is provider-native. Validate it before using it; the
    /// dispatcher's envelope checks do not authenticate or validate its contents.
    fn search(&self, _query: ValidatedSearch<'_>) -> Result<SearchPage> {
        bail!("this store does not support search")
    }

    /// Enumerate one page without a search predicate. A searchable store need not
    /// support enumeration; registering it must not initiate a full crawl.
    /// [`Discovery`] checks the advertised enumeration limit before calling this.
    /// Validate the native cursor and return `None` when there are no further pages.
    fn enumerate(&self, _page: ValidatedPage<'_>) -> Result<SearchPage> {
        bail!("this store does not support enumeration")
    }
}

/// Provider-defined filter fields and search limits, also usable to generate help.
/// For example, an adapter can expose an exact publishing-account filter:
///
/// ```
/// use cherenkov_model_data::discovery::*;
/// use std::collections::BTreeMap;
///
/// let schema = SearchSchema {
///     text_search: false,
///     max_page_size: 100,
///     fields: BTreeMap::from([("author".to_owned(), SearchField {
///         description: "Publishing account or organization".to_owned(),
///         value_type: ValueType::String,
///         operators: vec![FilterOperator::Equal],
///     })]),
/// };
/// ```
///
/// Return this schema in [`DiscoveryCapabilities::search`]. Field types and
/// operators validate operands; the adapter implements their matching behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSchema {
    /// Whether the provider accepts free-text queries.
    pub text_search: bool,
    /// Maximum number of results requested in one page.
    pub max_page_size: usize,
    /// Filter definitions keyed by the provider's field names, such as `author`.
    pub fields: BTreeMap<String, SearchField>,
}

/// Accepted operands and operations for one provider-specific search field.
/// Advertising a field does not imply every candidate has a known value for it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchField {
    /// Short explanation for CLI help, including units where relevant.
    pub description: String,
    /// Type of a filter operand, independent of the provider's stored representation.
    pub value_type: ValueType,
    /// Operations the provider supports for this field.
    pub operators: Vec<FilterOperator>,
}

/// JSON value types accepted as filter operands; values are never coerced.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ValueType {
    /// Text, including provider-defined identifiers and tags.
    String,
    /// A JSON boolean.
    Boolean,
    /// A signed or unsigned JSON integer, preserving its exact value.
    Integer,
    /// A nonnegative JSON integer, suitable for byte and parameter counts.
    Unsigned,
    /// A JSON number, including fractional values.
    Number,
    /// One of a provider's advertised string values.
    Choice {
        /// Case-sensitive accepted values.
        values: Vec<String>,
    },
}

/// Predicate operations; providers advertise only those they can honor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterOperator {
    /// Match an equal value.
    Equal,
    /// Match a known value that differs from the operand.
    NotEqual,
    /// Match a smaller value.
    Less,
    /// Match a value no greater than the operand.
    LessOrEqual,
    /// Match a larger value.
    Greater,
    /// Match a value no smaller than the operand.
    GreaterOrEqual,
    /// Provider-defined containment, such as a tag or substring match.
    Contains,
    /// Match any operand in a nonempty array of the field's advertised value type.
    AnyOf,
}

/// One predicate using a field from the selected store's search schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Filter {
    /// Provider-defined field key.
    pub field: String,
    /// Operation advertised for this field.
    pub operator: FilterOperator,
    /// Typed JSON operand, or an array of operands for [`FilterOperator::AnyOf`].
    pub value: Value,
}

/// Bounded page request shared by search and inventory enumeration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageRequest {
    /// Maximum results to return, greater than zero.
    pub limit: usize,
    /// Continuation from [`Discovery`] for the same store and unchanged request.
    /// Adapters receive their native cursor after the dispatcher checks its scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
}

impl PageRequest {
    /// Check the page size against the store's limit.
    pub fn validate(&self, maximum: usize) -> Result<()> {
        ensure!(
            self.limit > 0 && self.limit <= maximum,
            "page limit must be 1..={maximum}"
        );

        Ok(())
    }
}

/// Search text and predicates, all of which must match for a result to qualify.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchQuery {
    /// Free-text query, if the store supports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Provider-defined filters combined with AND.
    #[serde(default)]
    pub filters: Vec<Filter>,
    /// Result bound and continuation state.
    pub page: PageRequest,
}

/// A discovered candidate; the locator remains meaningful to its originating store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCandidate {
    /// Store-defined source locator that registration can resolve.
    /// A discovered branch or tag is not an immutable model identity.
    pub reference: String,
    /// Common descriptive facts, populated only when known.
    #[serde(default)]
    pub metadata: ModelMetadata,
    /// Provider-specific metadata. Missing or null values mean unknown.
    /// These keys need not coincide with the provider's search field names.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, Value>,
}

/// Optional descriptive metadata shared by discovery results.
/// Absence means unknown, rather than an empty string, zero, or a negative claim.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Publishing account or organization, not necessarily the model's author.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publisher: Option<String>,
    /// Provider-supplied model description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Declared architecture name, without implying engine support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// Reported license identifiers; absence makes no claim about licensing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub licenses: Option<Vec<String>>,
    /// Total parameter count, including inactive MoE experts when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<u64>,
}

/// Missing metadata that prevented evaluating a supported filter for candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataGap {
    /// Provider-defined filter field with missing values.
    pub field: String,
    /// Candidates omitted while producing this page, if the provider can count them.
    /// Counts for different fields may refer to the same candidates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates: Option<u64>,
}

/// One result page. An empty page can still carry a continuation cursor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPage<T = ModelCandidate> {
    /// Matching candidates, no more than the requested page limit.
    pub items: Vec<T>,
    /// Continuation for the same store and query; absence means no further page.
    pub next_cursor: Option<String>,
    /// Total matches across pages when known, not the number of items in this page.
    pub total: Option<u64>,
    /// Known gaps in filter metadata. Absence does not guarantee provider completeness.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gaps: Vec<MetadataGap>,
}

impl<T> Default for SearchPage<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            next_cursor: None,
            total: None,
            gaps: Vec::new(),
        }
    }
}
