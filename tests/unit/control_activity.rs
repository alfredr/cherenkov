use super::*;
use crate::qwen4_exp::gpu::{ExpertActivity, ExpertCounters};

fn populated_state() -> Arc<State> {
    let state = state();

    state.update(|s| {
        s.ready = true;
        s.memory.observed_at_uptime_seconds = 1.25;
        s.activity = ExpertActivity {
            layer_prefixes: vec!["model.layers.0".into(), "mtp.layers.0".into()],
            experts_per_layer: 3,
            layers: vec![crate::qwen4_exp::gpu::LayerStats::default(); 2],
            records: (1..=6)
                .map(|n| ExpertCounters {
                    selected_rows: n * 2,
                    cache_hits: n,
                    cache_misses: 1,
                    prefetch_requests: n,
                    read_requests: 1,
                    read_bytes_requested: 1024,
                })
                .collect(),
            ..Default::default()
        };
    });

    state
}

/// Stop the listener before removing its temporary socket directory.
struct SocketFixture {
    _listener: Listener,
    _directory: Directory,
    socket: PathBuf,
}

impl SocketFixture {
    fn new() -> Self {
        let directory = Directory::new();
        let socket = directory.0.join("control.sock");
        let listener = Listener::start(&socket, populated_state()).unwrap();

        Self {
            _listener: listener,
            _directory: directory,
            socket,
        }
    }
}

#[test]
fn socket_responses_round_trip_through_the_shared_stats_types() {
    use crate::control::stats::{Expert, Layer, Page, Snapshot, Summary};

    let fixture = SocketFixture::new();
    let socket = &fixture.socket;

    fn round_trip<T: serde::de::DeserializeOwned + serde::Serialize>(
        socket: &Path,
        command: Command,
    ) {
        let response = query(socket, command).unwrap();
        let decoded: T = serde_json::from_value(response.clone()).unwrap();

        assert_eq!(serde_json::to_value(decoded).unwrap(), response);
    }

    round_trip::<Snapshot<Summary>>(socket, Command::StatsSummary);
    round_trip::<Snapshot<Page<Layer>>>(
        socket,
        Command::StatsLayers {
            offset: 0,
            limit: 1,
        },
    );
    round_trip::<Snapshot<Page<Expert>>>(
        socket,
        Command::StatsExperts {
            layer: 0,
            offset: 0,
            limit: 1,
        },
    );
}

#[test]
fn publication_swaps_complete_snapshots_and_reuses_the_old_buffer() {
    let state = populated_state();
    let mut old_buffer = std::ptr::null();

    state.update(|s| old_buffer = s.activity.records.as_ptr());

    let mut spare = ExpertActivity::default();

    state.update(|s| spare = s.activity.clone());

    spare.records[0].selected_rows = 999;
    let prepared_buffer = spare.records.as_ptr();

    state.publish_observation(Default::default(), Default::default(), &mut spare);

    state.update(|s| assert_eq!(s.activity.records.as_ptr(), prepared_buffer));
    assert_eq!(spare.records.as_ptr(), old_buffer);

    spare.records[0].selected_rows = 0;

    assert_eq!(
        state.experts(0, 0, 1).unwrap()["data"][0]["selected_rows"],
        999
    );
}

#[test]
fn socket_layer_pages_aggregate_experts_without_bloating_status() {
    let fixture = SocketFixture::new();
    let socket = &fixture.socket;
    let layers = query(
        socket,
        Command::StatsLayers {
            offset: 0,
            limit: 1,
        },
    )
    .unwrap();

    assert_eq!(layers["total"], 2);
    assert_eq!(layers["next_offset"], 1);
    assert_eq!(layers["observed_at_uptime_seconds"], 1.25);
    assert_eq!(
        layers["data"],
        json!([{
            "layer": 0, "name": "model.layers.0", "experts": 3,
            "streaming": crate::qwen4_exp::gpu::LayerStats::default(),
            "selected_rows": 12, "cache_hits": 6, "cache_misses": 3,
            "prefetch_requests": 6, "read_requests": 3, "read_bytes_requested": 3072,
        }])
    );

    let status = query(socket, Command::Status).unwrap();

    assert!(status["stats"].get("activity").is_none());

    let summary = query(socket, Command::StatsSummary).unwrap();

    assert_eq!(summary["reads"]["requested_reads"], 0);
}

#[test]
fn socket_expert_pages_preserve_layer_and_expert_identity() {
    let fixture = SocketFixture::new();
    let socket = &fixture.socket;

    let experts = query(
        socket,
        Command::StatsExperts {
            layer: 1,
            offset: 1,
            limit: 2,
        },
    )
    .unwrap();

    assert_eq!(experts["offset"], 1);
    assert_eq!(experts["total"], 3);
    assert!(experts["next_offset"].is_null());
    assert_eq!(experts["data"][0]["layer"], 1);
    assert_eq!(experts["data"][0]["expert"], 1);
    assert_eq!(experts["data"][0]["selected_rows"], 10);
    assert_eq!(experts["data"][1]["expert"], 2);
    assert_eq!(experts["data"][1]["selected_rows"], 12);
}

#[test]
fn activity_pages_validate_bounds_and_allow_empty_final_page() {
    let state = populated_state();

    assert_eq!(state.layers(2, 64).unwrap()["data"], json!([]));
    assert_eq!(state.experts(0, 3, 64).unwrap()["data"], json!([]));
    assert!(state.layers(3, 64).is_err());
    assert!(state.experts(2, 0, 64).is_err());
    assert!(state.experts(0, 4, 64).is_err());

    for limit in [0, 129, usize::MAX] {
        assert!(state.layers(0, limit).is_err());
        assert!(state.experts(0, 0, limit).is_err());
    }
}

#[test]
fn maximum_expert_page_fits_control_frame() {
    let state = populated_state();

    state.update(|s| {
        s.activity.experts_per_layer = 128;
        s.activity.records = vec![
            ExpertCounters {
                selected_rows: u64::MAX,
                cache_hits: u64::MAX,
                cache_misses: u64::MAX,
                prefetch_requests: u64::MAX,
                read_requests: u64::MAX,
                read_bytes_requested: u64::MAX,
            };
            256
        ];
    });

    let page = state.experts(0, 0, 128).unwrap();
    let response = json!({"ok": true, "data": page, "error": null});

    assert!(serde_json::to_vec(&response).unwrap().len() < MAX_FRAME);
}

#[test]
fn summary_exports_rates_and_shared_phase_totals() {
    let state = populated_state();

    state.update(|s| {
        s.activity.elapsed_seconds = 2.0;
        s.activity.gpu_timestamps_available = true;
        let layer = &mut s.activity.layers[0];
        layer.quant[2].reads.demand.completed_bytes = 1000;
        layer.quant[2].reads.demand.completed_reads = 4;
        layer.quant[2].eligible_weak_misses = 2;
        layer.quant[2].cut_experts = 1;
        layer.phases.router_to_resident_seconds = 0.1;
        layer.phases.resident_to_fetched_seconds = 0.2;
        layer.phases.resident_seconds = 0.3;
        layer.phases.fetched_stage_seconds = 0.4;
        layer.phases.service_cpu_seconds = 0.05;
    });

    let summary = state.summary().unwrap();

    assert_eq!(summary["bytes_per_second"], 500.0);
    assert_eq!(summary["reads_per_second"], 2.0);
    assert_eq!(summary["streaming"]["quant"][2]["bits"], 2);
    assert_eq!(summary["streaming"]["quant"][2]["cut_experts"], 1);
    assert_eq!(summary["streaming"]["phases"]["service_cpu_seconds"], 0.05);
    assert_eq!(summary["gpu_stage_fraction"], 0.7);
}

#[test]
fn detailed_layer_pages_shrink_to_fit_and_continue_without_skipping() {
    let state = populated_state();

    state.update(|s| {
        s.activity.layer_prefixes = (0..128).map(|i| format!("model.layers.{i}")).collect();
        s.activity.records = vec![ExpertCounters::default(); 384];
        s.activity.layers = vec![crate::qwen4_exp::gpu::LayerStats::default(); 128];
    });

    let mut offset = 0;

    while offset < 128 {
        let page = state.layers(offset, 128).unwrap();

        assert!(serde_json::to_vec(&page).unwrap().len() < MAX_FRAME);

        let rows = page["data"].as_array().unwrap();

        assert!(!rows.is_empty());
        assert_eq!(rows[0]["layer"], offset);

        offset += rows.len();

        assert_eq!(rows.last().unwrap()["layer"], offset - 1);
        assert_eq!(
            page["next_offset"],
            (offset < 128)
                .then_some(offset)
                .map_or(Value::Null, Value::from)
        );
    }
}

#[test]
fn empty_summary_has_no_invented_rates_or_gpu_utilization() {
    let summary = populated_state().summary().unwrap();

    assert!(summary["bytes_per_second"].is_null());
    assert!(summary["gpu_stage_fraction"].is_null());
    assert_eq!(summary["gpu_timestamps_available"], false);
}

#[test]
fn stats_report_not_ready_until_a_snapshot_can_be_used() {
    let dir = Directory::new();
    let socket = dir.0.join("control.sock");
    let state = populated_state();

    state.update(|s| s.ready = false);

    let _listener = Listener::start(&socket, state.clone()).unwrap();

    for command in [
        Command::StatsSummary,
        Command::StatsLayers {
            offset: 0,
            limit: 1,
        },
        Command::StatsExperts {
            layer: 0,
            offset: 0,
            limit: 1,
        },
    ] {
        let error = query(&socket, command).unwrap_err();

        assert!(error.to_string().contains("not ready"), "{error}");
    }

    assert_eq!(
        query(&socket, Command::Status).unwrap()["stats"]["ready"],
        false
    );
    state.update(|s| s.ready = true);
    assert!(query(&socket, Command::StatsSummary).is_ok());
}

#[test]
fn inconsistent_activity_returns_errors_without_poisoning_the_stats_mutex() {
    let mutations: [fn(&mut ExpertActivity); 4] = [
        |a| {
            a.layer_prefixes.pop();
        },
        |a| {
            a.layers.pop();
        },
        |a| {
            a.records.pop();
        },
        |a| a.experts_per_layer = usize::MAX,
    ];

    for mutate in mutations {
        let state = populated_state();

        state.update(|s| mutate(&mut s.activity));
        assert!(state.summary().is_err());
        assert!(state.layers(0, 1).is_err());
        assert!(state.experts(0, 0, 1).is_err());
        assert_eq!(state.status()["stats"]["ready"], true);
    }
}

#[test]
fn stats_export_timer_initialization_failures() {
    let state = populated_state();

    state.update(|s| {
        s.activity.gpu_timing = crate::qwen4_exp::gpu::GpuTiming::Failed {
            error: "timestamp allocation failed".into(),
        }
    });

    let summary = state.summary().unwrap();

    assert_eq!(summary["gpu_timestamps_available"], false);
    assert_eq!(
        summary["gpu_timing"],
        json!({"status": "failed", "error": "timestamp allocation failed"})
    );
}

#[test]
fn summary_exports_prefill_chunks_alongside_completed_reads() {
    let state = populated_state();

    state.update(|s| {
        s.activity.prefill = crate::qwen4_exp::gpu::prefill::PrefillStats {
            chunks: 3,
            tokens: 275,
            min_chunk_tokens: 91,
            max_chunk_tokens: 92,
            seconds: 5.0,
            ..Default::default()
        };
        s.activity.layers[0].quant[2].reads.prefill.completed_bytes = 9_007_199_254_740_993;
    });

    let summary = state.summary().unwrap();

    assert_eq!(summary["prefill"]["chunks"], 3);
    assert_eq!(summary["prefill"]["tokens"], 275);
    assert_eq!(summary["prefill"]["min_chunk_tokens"], 91);
    assert_eq!(summary["prefill"]["max_chunk_tokens"], 92);
    assert_eq!(
        summary["reads"]["completed_bytes"],
        9_007_199_254_740_993_u64
    );
}
