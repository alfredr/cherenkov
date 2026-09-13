use super::*;

#[test]
fn model_source_references_reach_generation_and_subcommands() {
    let reference = "model:hf:example/model-q4:abcdef01";
    let cli = Cli::try_parse_from(["cherenkov", reference, "hello"]).unwrap();

    assert_eq!(
        cli.model_dir.as_deref(),
        Some(std::path::Path::new(reference))
    );

    for command in ["pack", "inspect"] {
        assert!(Cli::try_parse_from(["cherenkov", command, reference]).is_ok());
    }

    for command in ["show", "remove"] {
        assert!(Cli::try_parse_from(["cherenkov", "model", command, reference]).is_ok());
    }

    let cli = Cli::try_parse_from(["cherenkov", "serve", "--model", reference]).unwrap();
    let Some(Command::Serve(args)) = cli.command else {
        panic!("serve expected");
    };

    assert_eq!(args.overrides.model.as_deref(), Some(reference));
}

#[test]
fn pack_accepts_default_single_and_multiple_precisions() {
    for (args, expected) in [
        (vec![], vec![4]),
        (vec!["--experts", "3"], vec![3]),
        (vec!["--experts", "2,3"], vec![2, 3]),
        (vec!["--experts", "4", "3", "2"], vec![4, 3, 2]),
        (vec!["--experts", "2", "--experts", "3"], vec![2, 3]),
    ] {
        let cli =
            Cli::try_parse_from(["cherenkov", "pack", "/model"].into_iter().chain(args)).unwrap();
        let Some(Command::Prepare {
            experts, model_dir, ..
        }) = cli.command
        else {
            panic!("pack expected")
        };

        assert_eq!(experts, expected);
        assert_eq!(model_dir, Some(PathBuf::from("/model")));
    }
}

#[test]
fn pack_rejects_invalid_or_missing_precisions() {
    for value in ["1", "5", "2,5", "all", "-2", ""] {
        assert!(Cli::try_parse_from(["cherenkov", "pack", "--experts", value]).is_err());
    }

    assert!(Cli::try_parse_from(["cherenkov", "pack", "--experts"]).is_err());
}

#[test]
fn server_cli_records_only_explicit_overrides() {
    let cli = Cli::try_parse_from([
        "cherenkov",
        "serve",
        "/model",
        "--max-tokens",
        "64",
        "--port",
        "9090",
    ])
    .unwrap();
    let Some(Command::Serve(args)) = cli.command else {
        panic!("serve expected")
    };
    let root = tempfile::tempdir().unwrap();
    let source = args.source(Some(root.path().to_owned())).unwrap();

    assert_eq!(source.overrides.max_tokens, Some(64));
    assert_eq!(source.overrides.experts, None);
    assert_eq!(source.overrides.no_eos, None);
    assert_eq!(source.resolve().unwrap().server.port, 9090);
}

#[test]
fn control_commands_do_not_require_a_model_or_prompt() {
    for args in [
        vec!["cherenkov", "status", "--json"],
        vec!["cherenkov", "dash"],
        vec!["cherenkov", "config", "show"],
        vec!["cherenkov", "stats", "layers"],
        vec!["cherenkov", "stats", "summary"],
        vec![
            "cherenkov",
            "config",
            "reload",
            "--socket",
            "/private/socket",
        ],
        vec!["cherenkov", "serve", "--print-config"],
    ] {
        assert!(Cli::try_parse_from(args).is_ok());
    }
}

#[test]
fn dashboard_accepts_a_control_socket() {
    let cli = Cli::try_parse_from(["cherenkov", "dash", "--socket", "/private/socket"]).unwrap();

    assert!(
        matches!(cli.command, Some(Command::Dash { socket: Some(path) }) if path == std::path::Path::new("/private/socket"))
    );
}

#[test]
fn expert_stats_forward_page_and_socket_options() {
    let cli = Cli::try_parse_from([
        "cherenkov",
        "stats",
        "experts",
        "12",
        "--offset",
        "128",
        "--limit",
        "32",
        "--socket",
        "/private/socket",
    ])
    .unwrap();
    let Some(Command::Stats {
        target,
        socket,
        json,
    }) = cli.command
    else {
        panic!("stats expected")
    };

    assert_eq!(socket, Some(PathBuf::from("/private/socket")));
    assert!(!json);
    assert_eq!(
        serde_json::to_value(target.command()).unwrap(),
        serde_json::json!({
            "op": "stats_experts", "layer": 12, "offset": 128, "limit": 32,
        })
    );
}

#[test]
fn stats_accept_json_before_or_after_the_target() {
    for args in [
        vec!["stats", "--json", "summary"],
        vec!["stats", "summary", "--json"],
        vec!["stats", "layers", "--json"],
        vec!["stats", "experts", "0", "--json"],
    ] {
        let cli = Cli::try_parse_from(["cherenkov"].into_iter().chain(args)).unwrap();

        assert!(matches!(
            cli.command,
            Some(Command::Stats { json: true, .. })
        ));
    }
}

#[test]
fn stats_reject_unbounded_pages_and_missing_layer() {
    for limit in ["0", "129", "-1"] {
        assert!(Cli::try_parse_from(["cherenkov", "stats", "layers", "--limit", limit]).is_err());
    }

    assert!(Cli::try_parse_from(["cherenkov", "stats", "experts"]).is_err());
}

#[test]
fn server_overrides_preserve_explicit_zero_false_and_adaptive() {
    let cli = Cli::try_parse_from([
        "cherenkov",
        "serve",
        "--experts",
        "4",
        "--cut-weak",
        "0",
        "--drafts",
        "0",
        "--pool-gb",
        "adaptive",
        "--no-eos=false",
    ])
    .unwrap();
    let Some(Command::Serve(args)) = cli.command else {
        panic!("serve expected")
    };
    let overrides = args
        .source(Some(tempfile::tempdir().unwrap().path().to_owned()))
        .unwrap()
        .overrides;

    assert_eq!(overrides.experts, Some(4));
    assert_eq!(overrides.cut_weak, Some(0.0));
    assert_eq!(overrides.drafts, Some(0));
    assert_eq!(
        overrides.pool_gb,
        Some(cherenkov::options::PoolBudget::Adaptive)
    );
    assert_eq!(overrides.no_eos, Some(false));

    let cli = Cli::try_parse_from(["cherenkov", "serve", "--no-eos"]).unwrap();
    let Some(Command::Serve(args)) = cli.command else {
        panic!("serve expected")
    };

    assert_eq!(
        args.source(Some(tempfile::tempdir().unwrap().path().to_owned()))
            .unwrap()
            .overrides
            .no_eos,
        Some(true)
    );
}

#[test]
fn server_rejects_cli_only_options_and_invalid_numbers() {
    for args in [
        vec!["--raw"],
        vec!["--check"],
        vec!["--repeat", "1"],
        vec!["--experts", "1"],
        vec!["--drafts", "4"],
        vec!["--max-ctx", "0"],
        vec!["--max-tokens", "0"],
        vec!["--cut-weak", "NaN"],
        vec!["--pool-gb", "0"],
    ] {
        assert!(Cli::try_parse_from(["cherenkov", "serve"].into_iter().chain(args)).is_err());
    }
}

#[test]
fn portable_root_and_default_config_work_without_a_model_argument() {
    let dir = tempfile::tempdir().unwrap();

    std::fs::write(
        dir.path().join("cherenkov.toml"),
        "[defaults]\nmax_tokens=128",
    )
    .unwrap();

    let cli = Cli::try_parse_from([
        "cherenkov",
        "serve",
        "--root",
        dir.path().to_str().unwrap(),
        "--print-config",
    ])
    .unwrap();
    let Some(Command::Serve(args)) = cli.command else {
        panic!("serve expected")
    };
    let source = args.source(cli.root).unwrap();
    let config = source.resolve().unwrap();

    assert_eq!(config.defaults.max_tokens, 128);
    assert_eq!(config.server.root.as_deref(), Some(dir.path()));
    assert_eq!(
        config.model_dir().unwrap(),
        PathBuf::from(cherenkov::storage::default_model_reference())
    );

    for args in [
        vec!["paths"],
        vec!["download", "--metadata-only"],
        vec!["pack"],
    ] {
        assert!(Cli::try_parse_from(["cherenkov"].into_iter().chain(args)).is_ok());
    }
}

#[test]
fn indexed_commands_parse_without_a_model_path() {
    for args in [
        vec!["model", "list", "--json"],
        vec!["prepare", "hf://Example/Tiny@abcdef01", "--name", "tiny"],
        vec!["prepare", "disk://models/Example/Tiny", "--experts", "2,3"],
        vec!["store", "add", "models", "/models"],
        vec!["store", "add", "cache", "/cache", "--layout", "hf-cache"],
        vec!["store", "list", "--json"],
        vec!["store", "disable", "models"],
        vec!["serve", "--model", "tiny"],
        vec![
            "model",
            "add",
            "hf://example/tiny",
            "--revision",
            "main",
            "--name",
            "tiny",
        ],
        vec!["model", "show", "model:tiny"],
        vec!["model", "remove", "model:tiny", "--source-only"],
        vec!["model", "gc", "--dry-run", "--json"],
        vec!["inspect", "model:tiny", "--json"],
        vec!["pack", "model:tiny", "--experts", "4,3,2", "--keep-source"],
        vec!["serve", "--model", "model:tiny", "--print-config"],
    ] {
        assert!(Cli::try_parse_from(std::iter::once("cherenkov").chain(args)).is_ok());
    }

    assert!(Cli::try_parse_from(["cherenkov", "serve", "/path", "--model", "model:tiny"]).is_err());
}

#[test]
fn indexed_server_arguments_are_not_made_into_relative_paths() {
    let root = tempfile::tempdir().unwrap();
    let cli = Cli::try_parse_from(["cherenkov", "serve", "model:tiny"]).unwrap();
    let Some(Command::Serve(serve)) = cli.command else {
        panic!("serve expected")
    };
    let config = serve
        .source(Some(root.path().to_owned()))
        .unwrap()
        .resolve()
        .unwrap();

    assert_eq!(config.model_dir().unwrap(), PathBuf::from("model:tiny"));
}
