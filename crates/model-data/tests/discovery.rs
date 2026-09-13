use cherenkov_model_data::discovery::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[path = "discovery/dispatch.rs"]
mod dispatch;

/// Declare one filter independently of the store that will implement it.
fn schema(field: &str, value_type: ValueType, operators: &[FilterOperator]) -> SearchSchema {
    SearchSchema {
        text_search: true,
        max_page_size: 100,
        fields: BTreeMap::from([(
            field.to_owned(),
            SearchField {
                description: "Test filter".to_owned(),
                value_type,
                operators: operators.to_vec(),
            },
        )]),
    }
}

/// Build a first-page request with one predicate and no free-text constraint.
fn query(field: &str, operator: FilterOperator, value: Value) -> SearchQuery {
    SearchQuery {
        text: None,
        filters: vec![Filter {
            field: field.to_owned(),
            operator,
            value,
        }],
        page: PageRequest {
            limit: 10,
            cursor: None,
        },
    }
}

#[test]
fn providers_define_their_own_fields() {
    let remote = schema("author", ValueType::String, &[FilterOperator::Equal]);
    let local = schema("prepared", ValueType::Boolean, &[FilterOperator::Equal]);
    let by_author = query("author", FilterOperator::Equal, json!("example-team"));
    let by_prepared = query("prepared", FilterOperator::Equal, json!(true));

    assert!(remote.validate(&by_author).is_ok());
    assert!(local.validate(&by_prepared).is_ok());
    assert!(
        remote
            .validate(&by_prepared)
            .unwrap_err()
            .to_string()
            .contains("unsupported search field")
    );
    assert!(
        local
            .validate(&by_author)
            .unwrap_err()
            .to_string()
            .contains("unsupported search field")
    );
}

#[test]
fn operators_must_be_advertised_for_the_field() {
    let schema = schema("author", ValueType::String, &[FilterOperator::Equal]);
    let query = query("author", FilterOperator::Contains, json!("example"));

    assert!(
        schema
            .validate(&query)
            .unwrap_err()
            .to_string()
            .contains("unsupported operator")
    );
}

#[test]
fn operand_types_are_not_coerced() {
    let cases = [
        (
            ValueType::String,
            json!("text"),
            vec![json!(false), json!(1)],
        ),
        (
            ValueType::Boolean,
            json!(false),
            vec![json!("false"), json!(0)],
        ),
        (ValueType::Integer, json!(-1), vec![json!(1.5), json!("1")]),
        (ValueType::Unsigned, json!(0), vec![json!(-1), json!(1.5)]),
        (
            ValueType::Number,
            json!(1.5),
            vec![json!("1.5"), json!(true)],
        ),
        (
            ValueType::Choice {
                values: vec!["text-generation".to_owned()],
            },
            json!("text-generation"),
            vec![json!("Text-Generation"), json!("audio")],
        ),
    ];

    for (value_type, accepted, mut rejected) in cases {
        let schema = schema("field", value_type, &[FilterOperator::Equal]);

        assert!(
            schema
                .validate(&query("field", FilterOperator::Equal, accepted))
                .is_ok()
        );
        rejected.extend([Value::Null, json!([]), json!({})]);

        for value in rejected {
            let request = query("field", FilterOperator::Equal, value);

            assert!(schema.validate(&request).is_err(), "accepted {request:?}");
        }
    }
}

#[test]
fn any_of_requires_a_nonempty_array_of_typed_operands() {
    let schema = schema("tags", ValueType::String, &[FilterOperator::AnyOf]);

    assert!(
        schema
            .validate(&query("tags", FilterOperator::AnyOf, json!(["moe", "mlx"])))
            .is_ok()
    );

    for value in [
        json!([]),
        json!("moe"),
        json!(["moe", 1]),
        json!([null]),
        json!([["moe"]]),
    ] {
        assert!(
            schema
                .validate(&query("tags", FilterOperator::AnyOf, value))
                .is_err()
        );
    }
}

#[test]
fn missing_values_cannot_be_used_as_not_equal_operands() {
    let schema = schema("author", ValueType::String, &[FilterOperator::NotEqual]);

    assert!(
        schema
            .validate(&query("author", FilterOperator::NotEqual, Value::Null))
            .is_err()
    );
}

#[test]
fn page_and_text_limits_are_checked() {
    let mut schema = schema("author", ValueType::String, &[FilterOperator::Equal]);
    let mut query = query("author", FilterOperator::Equal, json!("example-team"));

    for limit in [0, 101, usize::MAX] {
        query.page.limit = limit;

        assert!(schema.validate(&query).is_err());
    }

    query.page.limit = 100;
    query.text = Some("small moe".to_owned());

    assert!(schema.validate(&query).is_ok());

    schema.text_search = false;

    assert!(
        schema
            .validate(&query)
            .unwrap_err()
            .to_string()
            .contains("text search")
    );

    query.text = None;

    assert!(schema.validate(&query).is_ok());
}

#[test]
fn parameter_counts_preserve_large_integers() {
    let schema = schema(
        "parameters",
        ValueType::Unsigned,
        &[FilterOperator::Greater],
    );
    let query = query("parameters", FilterOperator::Greater, json!(u64::MAX));
    let encoded = serde_json::to_string(&query).unwrap();
    let decoded: SearchQuery = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded.filters[0].value.as_u64(), Some(u64::MAX));
    assert!(schema.validate(&decoded).is_ok());
}

#[test]
fn result_metadata_preserves_absence_and_provider_fields() {
    let candidate: ModelCandidate = serde_json::from_value(json!({
        "reference": "hf://example-team/small-moe",
        "fields": { "gated": false, "card": { "tags": ["moe"] } }
    }))
    .unwrap();

    assert!(candidate.metadata.publisher.is_none());
    assert!(candidate.metadata.parameters.is_none());

    let encoded = serde_json::to_value(&candidate).unwrap();

    assert_eq!(encoded["metadata"], json!({}));
    assert_eq!(encoded["fields"]["gated"], json!(false));
    assert_eq!(encoded["fields"]["card"]["tags"], json!(["moe"]));

    let known_zero = ModelMetadata {
        parameters: Some(0),
        ..Default::default()
    };

    assert_eq!(
        serde_json::to_value(known_zero).unwrap(),
        json!({ "parameters": 0 })
    );
}

#[test]
fn empty_pages_can_continue_without_a_known_total() {
    let page: SearchPage = SearchPage {
        next_cursor: Some("opaque-continuation".to_owned()),
        gaps: vec![MetadataGap {
            field: "parameters".to_owned(),
            candidates: None,
        }],
        ..Default::default()
    };
    let encoded = serde_json::to_string(&page).unwrap();
    let decoded: SearchPage = serde_json::from_str(&encoded).unwrap();

    assert!(decoded.items.is_empty());
    assert_eq!(decoded.next_cursor.as_deref(), Some("opaque-continuation"));
    assert!(decoded.total.is_none());
    assert!(decoded.gaps[0].candidates.is_none());
}

/// An adapter with no discovery operations, exercising the default error paths.
struct UnsupportedStore;

impl StoreDiscovery for UnsupportedStore {
    fn capabilities(&self) -> DiscoveryCapabilities {
        DiscoveryCapabilities::default()
    }
}

#[test]
fn unsupported_discovery_returns_an_error_instead_of_an_empty_inventory() {
    let store = Discovery::new(StoreId("unsupported".to_owned()), &UnsupportedStore);
    let query = query("author", FilterOperator::Equal, json!("example-team"));

    assert!(store.capabilities().search.is_none());
    assert!(store.capabilities().enumeration_max_page_size.is_none());
    assert!(store.search(&query).is_err());
    assert!(store.enumerate(&query.page).is_err());
}

#[test]
fn request_typos_are_rejected_instead_of_broadening_the_search() {
    for input in [
        json!({"filter": [], "page": {"limit": 10}}),
        json!({"textt": "small", "page": {"limit": 10}}),
        json!({"page": {"limit": 10, "cursorr": "next"}}),
        json!({"filters": [{"field": "author", "operator": "equal", "value": "team", "extra": true}], "page": {"limit": 10}}),
    ] {
        let error = serde_json::from_value::<SearchQuery>(input).unwrap_err();

        assert!(error.to_string().contains("unknown field"), "{error}");
    }
}
