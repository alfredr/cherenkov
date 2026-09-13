use super::*;
use cherenkov::control::stats as wire;
use cherenkov::qwen4_exp::gpu::{ExpertCounters, LayerStats};
use serde_json::json;

#[path = "cli_output/terminal.rs"]
mod terminal_rendering;

fn observation() -> wire::Observation {
    wire::Observation {
        observed_at_uptime_seconds: 3.0,
        elapsed_seconds: 2.0,
        gpu_timestamps_available: false,
        gpu_timing: cherenkov::qwen4_exp::gpu::GpuTiming::Unsupported,
    }
}

pub(super) fn summary() -> Value {
    let mut streaming = LayerStats::default();
    streaming.quant[2].reads.demand.completed_reads = 4;
    streaming.quant[2].reads.demand.completed_bytes = 1000;

    serde_json::to_value(wire::Snapshot {
        observation: observation(),
        data: wire::Summary {
            prefill: Default::default(),
            reads: streaming.reads(),
            streaming,
            rates: wire::Rates {
                bytes_per_second: Some(500.0),
                reads_per_second: Some(2.0),
                gpu_handoff_gap_fraction: None,
                gpu_stage_fraction: None,
            },
        },
    })
    .unwrap()
}

fn page(data: Value, next_offset: Option<usize>) -> Value {
    serde_json::to_value(wire::Snapshot {
        observation: observation(),
        data: wire::Page {
            offset: 0,
            total: 2,
            next_offset,
            data: data.as_array().unwrap().clone(),
        },
    })
    .unwrap()
}

/// Render the Markdown view while leaving response construction in each test.
fn markdown(command: &Command, response: &Value) -> String {
    stats::View::from_response(command, response)
        .unwrap()
        .report()
        .markdown()
}

#[test]
fn summary_uses_engine_schema_and_marks_unavailable_rates() {
    let response = summary();
    let markdown = markdown(&Command::StatsSummary, &response);

    assert!(markdown.contains("500.000"));
    assert!(markdown.contains("n/a"));
    assert!(markdown.contains("GPU stage coverage (%)"));
    assert!(markdown.contains("| Metric"));
    assert!(!markdown.contains('\u{1b}'));
}

#[test]
fn pages_preserve_large_counters_and_show_next_offset() {
    let mut expert = serde_json::to_value(ExpertCounters {
        selected_rows: u64::MAX,
        ..Default::default()
    })
    .unwrap();
    expert["layer"] = json!(0);
    expert["expert"] = json!(1);
    let response = page(json!([expert]), Some(1));
    let command = Command::StatsExperts {
        layer: 0,
        offset: 0,
        limit: 1,
    };
    let markdown = markdown(&command, &response);

    assert!(markdown.contains(&u64::MAX.to_string()));
    assert!(markdown.contains("Next page: --offset 1"));
}

#[test]
fn empty_final_page_has_no_next_page_instruction() {
    let command = Command::StatsLayers {
        offset: 0,
        limit: 1,
    };
    let markdown = markdown(&command, &page(json!([]), None));

    assert!(markdown.contains("0 entries"));
    assert!(!markdown.contains("Next page"));
}

#[test]
fn malformed_stats_do_not_turn_into_zero_counters() {
    assert!(stats::View::from_response(&Command::StatsSummary, &json!({})).is_err());
}

#[test]
fn summary_explains_why_gpu_timing_is_unavailable() {
    let mut response = summary();
    response["gpu_timing"] = json!({"status": "failed", "error": "timestamp allocation failed"});
    let markdown = format_stats(&Command::StatsSummary, &response, false).unwrap();

    assert!(markdown.contains("failed: timestamp allocation failed"));
}

#[test]
fn json_retains_full_response_including_extensions_and_large_integers() {
    let mut response = summary();
    response["extension"] = json!({"counter": u64::MAX});
    let output = format_stats(&Command::StatsSummary, &response, true).unwrap();

    assert_eq!(serde_json::from_str::<Value>(&output).unwrap(), response);
    assert!(!output.contains('\u{1b}'));
}

#[test]
fn layer_names_cannot_add_table_columns_or_terminal_escapes() {
    let mut layer = serde_json::to_value(ExpertCounters::default()).unwrap();
    layer["layer"] = json!(0);
    layer["experts"] = json!(1);
    layer["streaming"] = json!(LayerStats::default());
    layer["name"] = json!("model.shared_expert|\n\u{1b}[31m");
    let command = Command::StatsLayers {
        offset: 0,
        limit: 1,
    };
    let markdown = format_stats(&command, &page(json!([layer]), None), false).unwrap();
    let row = markdown
        .lines()
        .find(|line| line.contains("model.shared_expert"))
        .unwrap();

    assert_eq!(row.matches('|').count(), 9);
    assert!(!markdown.contains('\u{1b}'));
}

#[test]
fn prefill_chunks_have_a_text_view_and_accept_older_snapshots() {
    let mut data = summary();
    data["prefill"]["max_chunk_tokens"] = json!(1024);
    let text = format_stats(&Command::StatsSummary, &data, false).unwrap();

    assert!(text.contains("Prefill chunks"));
    assert!(text.contains("Largest chunk (tokens)"));
    assert!(text.contains("1024"));

    data.as_object_mut().unwrap().remove("prefill");
    assert!(format_stats(&Command::StatsSummary, &data, false).is_ok());
}
