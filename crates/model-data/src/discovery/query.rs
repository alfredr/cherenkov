//! Schema validation checks operands; adapters evaluate them against metadata.

use super::{FilterOperator, SearchQuery, SearchSchema, ValueType};
use anyhow::{Context, Result, ensure};
use serde_json::Value;

impl SearchSchema {
    /// Reject unsupported text, unknown fields, unsupported operators, and invalid
    /// operands. Validation does not perform I/O or interpret provider metadata.
    pub fn validate(&self, query: &SearchQuery) -> Result<()> {
        query.page.validate(self.max_page_size)?;
        ensure!(
            self.text_search || query.text.is_none(),
            "this store does not support text search"
        );

        for filter in &query.filters {
            let field = self
                .fields
                .get(&filter.field)
                .with_context(|| format!("unsupported search field {:?}", filter.field))?;

            ensure!(
                field.operators.contains(&filter.operator),
                "unsupported operator {:?} for search field {:?}",
                filter.operator,
                filter.field
            );
            ensure!(
                field.value_type.accepts(&filter.value, filter.operator),
                "invalid value for search field {:?}: expected {:?}",
                filter.field,
                field.value_type
            );
        }

        Ok(())
    }
}

impl ValueType {
    /// Apply the field's scalar type to one operand or every `AnyOf` element.
    fn accepts(&self, value: &Value, operator: FilterOperator) -> bool {
        if operator != FilterOperator::AnyOf {
            return self.accepts_scalar(value);
        }

        value.as_array().is_some_and(|values| {
            !values.is_empty() && values.iter().all(|value| self.accepts_scalar(value))
        })
    }

    /// Check JSON representation without coercion or loss of integer precision.
    fn accepts_scalar(&self, value: &Value) -> bool {
        match self {
            Self::String => value.is_string(),
            Self::Boolean => value.is_boolean(),
            Self::Integer => value.is_i64() || value.is_u64(),
            Self::Unsigned => value.is_u64(),
            Self::Number => value.is_number(),
            Self::Choice { values } => value
                .as_str()
                .is_some_and(|text| values.iter().any(|v| v == text)),
        }
    }
}
