//! Observations from one model operation, independent of terminal rendering.

use serde::Serialize;

/// Progress during remote model inspection. Values are owned so observers can
/// forward events to a channel. The operation's `Result` signals completion;
/// these events do not establish success or describe durable index state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelEvent {
    /// Emitted before the first remote metadata request.
    Resolving { source: String },
    /// Header counts start at zero and advance after each successful shard read.
    /// `file` identifies the latest completed shard and is absent initially.
    Headers {
        completed: usize,
        total: usize,
        file: Option<String>,
    },
    /// Headers are complete; architecture metadata may require small range reads.
    ReadingMetadata,
}
