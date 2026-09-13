use super::{query, schema};
use anyhow::{Result, bail, ensure};
use cherenkov_model_data::discovery::*;
use serde_json::{Value, json};
use std::cell::Cell;

/// Paged metadata fixture with known, missing, and explicit-null author values.
/// The read counter exposes whether rejected requests reached storage.
struct MemoryStore {
    rows: Vec<ModelCandidate>,
    capabilities: DiscoveryCapabilities,
    reads: Cell<usize>,
}

impl MemoryStore {
    /// Build an inventory whose first page can contain only unknown metadata.
    fn new() -> Self {
        Self {
            rows: [
                None,
                Some(Value::Null),
                Some(json!("alpha")),
                Some(json!("beta")),
            ]
            .into_iter()
            .enumerate()
            .map(|(index, author)| ModelCandidate {
                reference: format!("model-{index}"),
                metadata: ModelMetadata::default(),
                fields: author
                    .map(|value| ("author".to_owned(), value))
                    .into_iter()
                    .collect(),
            })
            .collect(),
            capabilities: DiscoveryCapabilities {
                enumeration_max_page_size: Some(4),
                search: Some(SearchSchema {
                    text_search: false,
                    ..schema(
                        "author",
                        ValueType::String,
                        &[FilterOperator::Equal, FilterOperator::NotEqual],
                    )
                }),
            },
            reads: Cell::new(0),
        }
    }

    /// Read a bounded input page using a row offset as the provider-native cursor.
    fn read_page(&self, request: &PageRequest) -> Result<SearchPage> {
        let start = request.cursor.as_deref().unwrap_or("0").parse::<usize>()?;

        ensure!(start <= self.rows.len(), "invalid native cursor");
        self.reads.set(self.reads.get() + 1);

        let end = start.saturating_add(request.limit).min(self.rows.len());

        Ok(SearchPage {
            items: self.rows[start..end].to_vec(),
            next_cursor: (end < self.rows.len()).then(|| end.to_string()),
            ..Default::default()
        })
    }
}

impl StoreDiscovery for MemoryStore {
    fn capabilities(&self) -> DiscoveryCapabilities {
        self.capabilities.clone()
    }

    fn search(&self, request: ValidatedSearch<'_>) -> Result<SearchPage> {
        let query = request.query();
        let mut page = self.read_page(&query.page)?;
        let mut matches = Vec::new();

        for candidate in page.items {
            if matches_filters(&candidate, &query.filters, &mut page.gaps)? {
                matches.push(candidate);
            }
        }

        page.items = matches;

        Ok(page)
    }

    fn enumerate(&self, request: ValidatedPage<'_>) -> Result<SearchPage> {
        self.read_page(request.page())
    }
}

/// Reference equality semantics for adapter tests, reporting unknown values as gaps.
fn matches_filters(
    candidate: &ModelCandidate,
    filters: &[Filter],
    gaps: &mut Vec<MetadataGap>,
) -> Result<bool> {
    for filter in filters {
        let Some(value) = candidate
            .fields
            .get(&filter.field)
            .filter(|value| !value.is_null())
        else {
            gaps.push(MetadataGap {
                field: filter.field.clone(),
                candidates: Some(1),
            });

            return Ok(false);
        };
        let matches = match filter.operator {
            FilterOperator::Equal => value == &filter.value,
            FilterOperator::NotEqual => value != &filter.value,
            _ => bail!("unexpected operator"),
        };

        if !matches {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Bind a fixture to an explicit origin so tests can distinguish identical stores.
fn discovery<'a>(id: &str, store: &'a MemoryStore) -> Discovery<'a> {
    Discovery::new(StoreId(id.to_owned()), store)
}

/// Query all four input rows, including the two with unknown author metadata.
fn author_query(operator: FilterOperator) -> SearchQuery {
    let mut request = query("author", operator, json!("alpha"));

    request.page.limit = 4;

    request
}

#[test]
fn invalid_requests_never_reach_the_adapter() {
    let store = MemoryStore::new();
    let dispatcher = discovery("local", &store);
    let mut invalid = vec![
        query("missing", FilterOperator::Equal, json!("alpha")),
        query("author", FilterOperator::Contains, json!("alpha")),
        query("author", FilterOperator::Equal, json!(false)),
    ];
    let mut text = author_query(FilterOperator::Equal);

    text.text = Some("unsupported".to_owned());

    invalid.push(text);

    for limit in [0, 101] {
        let mut request = author_query(FilterOperator::Equal);

        request.page.limit = limit;

        invalid.push(request);
    }

    for request in invalid {
        assert!(dispatcher.search(&request).is_err());
    }

    for limit in [0, 5] {
        assert!(
            dispatcher
                .enumerate(&PageRequest {
                    limit,
                    cursor: None
                })
                .is_err()
        );
    }

    assert_eq!(store.reads.get(), 0);
}

#[test]
fn capability_checks_happen_before_dispatch() {
    let mut store = MemoryStore::new();

    store.capabilities = DiscoveryCapabilities::default();

    let dispatcher = discovery("disabled", &store);
    let request = author_query(FilterOperator::Equal);

    assert!(dispatcher.search(&request).is_err());
    assert!(dispatcher.enumerate(&request.page).is_err());
    assert_eq!(store.reads.get(), 0);
}

#[test]
fn search_does_not_require_enumeration_support() {
    let mut store = MemoryStore::new();

    store.capabilities.enumeration_max_page_size = None;

    let dispatcher = discovery("remote", &store);
    let request = author_query(FilterOperator::Equal);

    assert_eq!(dispatcher.search(&request).unwrap().items.len(), 1);
    assert!(dispatcher.enumerate(&request.page).is_err());
    assert_eq!(store.reads.get(), 1);
}

#[test]
fn missing_and_null_metadata_never_match_even_for_not_equal() {
    let store = MemoryStore::new();
    let dispatcher = discovery("local", &store);

    for (operator, expected) in [
        (FilterOperator::Equal, "model-2"),
        (FilterOperator::NotEqual, "model-3"),
    ] {
        let page = dispatcher.search(&author_query(operator)).unwrap();

        assert_eq!(page.items.len(), 1);
        assert_eq!(page.items[0].candidate.reference, expected);
        assert_eq!(
            page.gaps
                .iter()
                .filter_map(|gap| gap.candidates)
                .sum::<u64>(),
            2
        );
    }
}

#[test]
fn empty_filtered_page_can_resume_after_serialization() {
    let store = MemoryStore::new();
    let dispatcher = discovery("local", &store);
    let mut request = author_query(FilterOperator::NotEqual);

    request.page.limit = 2;

    let first = dispatcher.search(&request).unwrap();

    assert!(first.items.is_empty());
    assert!(first.next_cursor.is_some());

    let serialized = serde_json::to_string(&first).unwrap();
    let decoded: SearchPage<StoredCandidate> = serde_json::from_str(&serialized).unwrap();

    request.page.cursor = decoded.next_cursor;

    let second = dispatcher.search(&request).unwrap();

    assert_eq!(second.items.len(), 1);
    assert_eq!(second.items[0].candidate.reference, "model-3");
    assert!(second.next_cursor.is_none());
    assert_eq!(store.reads.get(), 2);
}

#[test]
fn cursors_cannot_be_reused_for_another_store_or_request() {
    let store = MemoryStore::new();
    let dispatcher = discovery("first", &store);
    let mut request = author_query(FilterOperator::Equal);

    request.page.limit = 2;
    request.page.cursor = dispatcher.search(&request).unwrap().next_cursor;

    assert!(discovery("second", &store).search(&request).is_err());
    assert!(dispatcher.enumerate(&request.page).is_err());

    let mut changed = request.clone();

    changed.page.limit = 3;

    assert!(dispatcher.search(&changed).is_err());

    changed = request.clone();
    changed.filters[0].value = json!("beta");

    assert!(dispatcher.search(&changed).is_err());

    changed.page.cursor = Some("malformed".to_owned());

    assert!(dispatcher.search(&changed).is_err());
    assert_eq!(store.reads.get(), 1);
}

#[test]
fn enumeration_continuation_preserves_the_inventory() {
    let store = MemoryStore::new();
    let dispatcher = discovery("local", &store);
    let mut request = PageRequest {
        limit: 2,
        cursor: None,
    };
    let mut items = Vec::new();

    loop {
        let page = dispatcher.enumerate(&request).unwrap();

        items.extend(page.items.into_iter().map(|item| item.candidate.reference));

        request.cursor = page.next_cursor;

        if request.cursor.is_none() {
            break;
        }

        assert!(store.reads.get() < 3, "enumeration did not advance");
    }

    assert_eq!(items, ["model-0", "model-1", "model-2", "model-3"]);
}

#[test]
fn identical_references_from_distinct_stores_retain_their_origin() {
    let store = MemoryStore::new();
    let request = author_query(FilterOperator::Equal);
    let mut combined = discovery("first", &store).search(&request).unwrap().items;

    combined.extend(discovery("second", &store).search(&request).unwrap().items);

    let serialized = serde_json::to_string(&combined).unwrap();
    let decoded: Vec<StoredCandidate> = serde_json::from_str(&serialized).unwrap();

    assert_eq!(
        decoded[0].candidate.reference,
        decoded[1].candidate.reference
    );
    assert_eq!(decoded[0].store.0, "first");
    assert_eq!(decoded[1].store.0, "second");
}

/// Deliberate adapter contract violations exercised by dispatcher response checks.
enum BadResponse {
    Oversized,
    RepeatedCursor,
}

impl StoreDiscovery for BadResponse {
    fn capabilities(&self) -> DiscoveryCapabilities {
        MemoryStore::new().capabilities
    }

    fn search(&self, _request: ValidatedSearch<'_>) -> Result<SearchPage> {
        match self {
            Self::Oversized => Ok(SearchPage {
                items: MemoryStore::new().rows,
                ..Default::default()
            }),
            Self::RepeatedCursor => Ok(SearchPage {
                next_cursor: Some("same-position".to_owned()),
                ..Default::default()
            }),
        }
    }
}

#[test]
fn oversized_adapter_pages_are_rejected() {
    let dispatcher = Discovery::new(StoreId("bad".to_owned()), &BadResponse::Oversized);
    let mut request = author_query(FilterOperator::Equal);

    request.page.limit = 1;

    let error = dispatcher.search(&request).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("exceeded the requested page limit")
    );
}

#[test]
fn a_repeated_native_cursor_cannot_cause_an_endless_walk() {
    let dispatcher = Discovery::new(StoreId("bad".to_owned()), &BadResponse::RepeatedCursor);
    let mut request = author_query(FilterOperator::Equal);

    request.page.cursor = dispatcher.search(&request).unwrap().next_cursor;

    let error = dispatcher.search(&request).unwrap_err();

    assert!(error.to_string().contains("did not advance"));
}

#[test]
fn numeric_filters_resume_without_reparsing_their_values() {
    let mut store = MemoryStore::new();

    store.capabilities.search = Some(schema(
        "score",
        ValueType::Number,
        &[FilterOperator::Greater],
    ));

    let dispatcher = discovery("numeric", &store);

    // The first value changes by one ULP when reparsed without float_roundtrip.
    for value in [
        json!(1.5807643528582219e120),
        json!(0.1),
        json!(u64::MAX),
        json!(i64::MIN),
    ] {
        let mut request = query("score", FilterOperator::Greater, value);

        request.page.limit = 2;

        let first = dispatcher.search(&request).unwrap();
        let encoded = serde_json::to_string(&first).unwrap();
        let decoded: SearchPage<StoredCandidate> = serde_json::from_str(&encoded).unwrap();

        request.page.cursor = decoded.next_cursor;

        assert!(request.page.cursor.is_some());
        assert!(dispatcher.search(&request).unwrap().next_cursor.is_none());
    }
}
