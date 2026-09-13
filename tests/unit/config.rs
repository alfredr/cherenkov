use super::*;

#[test]
fn defaults_preserve_engine_choices_and_toml_roundtrips() {
    let c = Config::default();

    c.validate().unwrap();

    let o = c.options();

    assert_eq!(
        (o.experts, o.miss_bits(), o.drafts, o.max_ctx, o.max_tokens),
        (4, 4, 2, 2048, 64)
    );
    assert_eq!(
        toml::from_str::<Config>(&toml::to_string_pretty(&c).unwrap()).unwrap(),
        c
    );
}

#[test]
fn documented_example_resolves_to_current_defaults() {
    let c: Config = toml::from_str(include_str!("../../cherenkov.example.toml")).unwrap();

    c.validate().unwrap();
    assert_eq!(c, Config::default());
}

#[test]
fn typed_policy_rejects_invalid_or_unimplemented_fields() {
    for text in [
        "[defaults]\ntemperature = 0.7",
        "[limits]\nmax_sessions = 1025",
        "[experts]\nresident_bits = 1",
        "[experts]\nresident_bits = 3\nmiss_bits = 2",
        "[experts]\npool_gb = 'unknown'",
        "[limits]\nmemory_gb = nan",
        "[limits]\nmemory_gb = 1e100",
        "[limits]\nqueued_requests = 0",
        "[limits]\nprefix_cache_mib = 2049",
        "[limits]\nmax_output_tokens = 10",
    ] {
        assert!(
            toml::from_str::<Config>(text)
                .and_then(|c| c.validate().map(|_| c).map_err(serde::de::Error::custom))
                .is_err(),
            "{text}"
        );
    }

    let c: Config =
        toml::from_str("[experts]\nresident_bits = 4\nmiss_bits = 2\nbuild_missing_store = false")
            .unwrap();

    c.validate().unwrap();
    assert_eq!(c.options().miss_bits(), 2);
    assert!(!c.options().build_missing_store);
}

#[test]
fn explicit_cli_values_override_files_without_clobbering_other_fields() {
    let mut c: Config =
        toml::from_str("[experts]\nresident_bits = 3\n[defaults]\nmax_tokens = 128\nstream = true")
            .unwrap();
    let overrides = Overrides {
        max_tokens: Some(64),
        ..Overrides::default()
    };

    overrides.apply(&mut c);
    assert_eq!(c.defaults.max_tokens, 64);
    assert!(c.defaults.stream);
    assert_eq!(c.experts.resident_bits, 3);
}

#[test]
fn only_request_defaults_can_reload() {
    let c = Config::default();
    let mut next = c.clone();
    next.defaults.stream = true;

    assert!(c.restart_changes(&next).is_empty());

    next.limits.context_tokens = 4096;
    next.experts.miss_bits = Some(2);

    assert_eq!(c.restart_changes(&next), vec!["limits", "experts"]);
}

#[test]
fn pool_budget_modes_and_numbers_roundtrip() {
    for (value, expected) in [
        ("'adaptive'", PoolBudget::Adaptive),
        ("'max'", PoolBudget::Max),
        ("12", PoolBudget::Gb(12.0)),
        ("12.5", PoolBudget::Gb(12.5)),
    ] {
        let config: Config = toml::from_str(&format!("[experts]\npool_gb = {value}")).unwrap();

        config.validate().unwrap();
        assert_eq!(config.options().pool_gb, expected);

        let encoded = toml::to_string(&config).unwrap();

        assert_eq!(toml::from_str::<Config>(&encoded).unwrap(), config);
    }

    for invalid in ["'unknown'", "0", "-1", "nan", "inf", "true"] {
        assert!(toml::from_str::<Config>(&format!("[experts]\npool_gb = {invalid}")).is_err());
    }
}

#[test]
fn explicit_defaults_reset_configured_values() {
    let mut config: Config = toml::from_str(
        "[server]\ndrafts = 3\n[experts]\nresident_bits = 3\ncut_weak = 0.08\npool_gb = 'max'\n[defaults]\nno_eos = true"
    ).unwrap();

    Overrides {
        experts: Some(4),
        cut_weak: Some(0.0),
        drafts: Some(0),
        pool_gb: Some(PoolBudget::Adaptive),
        no_eos: Some(false),
        ..Overrides::default()
    }
    .apply(&mut config);
    config.validate().unwrap();
    assert_eq!(config.experts, Experts::default());
    assert_eq!(config.server.drafts, 0);
    assert!(!config.defaults.no_eos);
}

#[test]
fn roots_resolve_relative_to_config_and_cli_overrides_survive_reload() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("settings.toml");

    std::fs::write(&file, "[server]\nroot='data'\nmodel_dir='checkpoint'").unwrap();

    let mut source = Source {
        path: Some(file),
        overrides: Overrides::default(),
    };
    let config = source.resolve().unwrap();

    assert_eq!(
        config.server.root.as_deref(),
        Some(dir.path().join("data").as_path())
    );
    assert_eq!(config.model_dir().unwrap(), dir.path().join("checkpoint"));

    source.overrides.root = Some(dir.path().join("override"));
    let updated = source.resolve().unwrap();

    assert_eq!(updated.paths().unwrap().data, dir.path().join("override"));
    assert_eq!(updated.model_dir().unwrap(), config.model_dir().unwrap());
    assert_eq!(config.restart_changes(&updated), vec!["server"]);
}

#[test]
fn larger_memory_budgets_and_quanta_are_explicit_options() {
    let defaults = Config::default();

    assert_eq!(defaults.limits.memory_gb, 25.0);
    assert_eq!(defaults.limits.prefill_quantum, 128);
    assert_eq!(defaults.limits.prefill_chunk_seconds, 2.0);

    for gb in [26.0, 40.0, 96.0] {
        for quantum in [128, 1024, 4096] {
            let text = format!("[limits]\nmemory_gb = {gb}\nprefill_quantum = {quantum}");
            let config: Config = toml::from_str(&text).unwrap();

            config.validate().unwrap();
            assert_eq!(config.memory_bytes(), (gb * BYTES_PER_GB as f64) as usize);
            assert_eq!(config.restart_changes(&defaults), vec!["limits"]);
        }
    }

    for quantum in [0, 4097, usize::MAX] {
        let mut config = defaults.clone();
        config.limits.prefill_quantum = quantum;

        assert!(config.validate().is_err());
    }

    for seconds in [-1.0, 60.5, f64::NAN, f64::INFINITY] {
        let mut config = defaults.clone();
        config.limits.prefill_chunk_seconds = seconds;

        assert!(config.validate().is_err());
    }
}

#[test]
fn index_model_selection_has_explicit_precedence_and_requires_restart() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("server.toml");

    std::fs::write(&path, "[server]\nmodel = 'model:tiny'\n").unwrap();

    let source = Source {
        path: Some(path.clone()),
        overrides: Overrides::default(),
    };
    let original = source.resolve().unwrap();

    assert_eq!(
        original.model_dir().unwrap(),
        std::path::PathBuf::from("model:tiny")
    );

    let cli = Source {
        path: Some(path.clone()),
        overrides: Overrides {
            model_dir: Some(dir.path().join("local")),
            ..Default::default()
        },
    }
    .resolve()
    .unwrap();

    assert!(cli.server.model.is_none());
    assert!(original.restart_changes(&cli).contains(&"server"));
    std::fs::write(&path, "[server]\nmodel_dir = 'local'\n").unwrap();

    let cli = Source {
        path: Some(path.clone()),
        overrides: Overrides {
            model: Some("model:second".into()),
            ..Default::default()
        },
    }
    .resolve()
    .unwrap();

    assert!(cli.server.model_dir.is_none());
    std::fs::write(
        &path,
        "[server]\nmodel = 'model:tiny'\nmodel_dir = 'local'\n",
    )
    .unwrap();
    assert!(source.resolve().is_err());
}

#[test]
fn model_selectors_preserve_urls_and_anchor_only_explicit_paths() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config.toml");
    let source = Source {
        path: Some(path.clone()),
        overrides: Overrides::default(),
    };

    for model in ["tiny", "hf://Qwen/Tiny@abcdef01", "disk://models/Qwen/Tiny"] {
        std::fs::write(&path, format!("[server]\nmodel = '{model}'\n")).unwrap();
        assert_eq!(
            source.resolve().unwrap().server.model.as_deref(),
            Some(model)
        );
    }

    std::fs::write(&path, "[server]\nmodel = './models/tiny'\n").unwrap();
    assert_eq!(
        source.resolve().unwrap().model_dir().unwrap(),
        dir.path().join("./models/tiny")
    );

    std::fs::write(&path, "[server]\nmodel = 'https://example.com/tiny'\n").unwrap();
    assert!(source.resolve().is_err());
}
