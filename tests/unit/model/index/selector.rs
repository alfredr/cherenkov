use super::*;

#[test]
fn urls_map_to_source_fields_without_losing_case_or_revision() {
    for (input, expected) in [
        (
            "hf://Qwen/Model-Q4@abcdef01",
            Selector::Hub {
                repo: "Qwen/Model-Q4".into(),
                revision: Some("abcdef01".into()),
            },
        ),
        (
            "disk://Models/Qwen/Model-Q4@main",
            Selector::Disk {
                store: "Models".into(),
                repo: "Qwen/Model-Q4".into(),
                revision: Some("main".into()),
            },
        ),
        (
            "hf://Qwen/Model@feature/branch",
            Selector::Hub {
                repo: "Qwen/Model".into(),
                revision: Some("feature/branch".into()),
            },
        ),
        (
            "hf://Other/Model-Q4",
            Selector::Hub {
                repo: "Other/Model-Q4".into(),
                revision: None,
            },
        ),
    ] {
        assert_eq!(parse(Path::new(input)).unwrap(), expected, "{input}");
    }
}

#[test]
fn malformed_sources_are_not_reinterpreted_as_aliases_or_paths() {
    for input in [
        "hf:Qwen/Model",
        "hf://Qwen",
        "hf://Qwen/Model/extra",
        "hf://user:pass@Qwen/Model",
        "hf://Qwen:80/Model",
        "hf://Qwen/Model?token=x",
        "hf://Qwen/Model#ref",
        "hf://Qwen/Model@",
        "hf://Qwen/Model@main@extra",
        "hf://Qwen/../Model",
        "hf://Qwen/%2e%2e/Model",
        "hf://Qwen/Model@a b",
        "hf://Qwen/Model@/branch",
        "disk://models/Qwen",
        "disk://models/Qwen/Model/extra",
        "disk[models]://Qwen/Model",
        "https://example.com/model",
        "model://alias",
        "model:",
        "model:hf:bad:aaaaaaaa",
    ] {
        assert!(parse(Path::new(input)).is_err(), "accepted {input}");
    }
}

#[test]
fn aliases_and_explicit_paths_stay_distinct() {
    for alias in ["tiny", ".tiny", "abcdef01abcdef01abcdef01abcdef01"] {
        assert_eq!(
            parse(Path::new(alias)).unwrap(),
            Selector::Registered(alias.into())
        );
    }

    for path in [
        "./tiny",
        "../tiny",
        "/models/model:name",
        "./hf://owner/repo",
        "models/tiny",
    ] {
        assert_eq!(parse(Path::new(path)).unwrap(), Selector::Path(path.into()));
    }
}

#[test]
fn legacy_forms_map_to_the_same_types() {
    assert_eq!(
        parse(Path::new("model:tiny")).unwrap(),
        parse(Path::new("tiny")).unwrap()
    );
    assert_eq!(
        parse(Path::new("model:hf:Qwen/Model:abcdef01")).unwrap(),
        parse(Path::new("hf://Qwen/Model@abcdef01")).unwrap()
    );
    assert_eq!(
        parse(Path::new("model:local:/models/local:name:abcdef01")).unwrap(),
        Selector::LocalRevision {
            path: "/models/local:name".into(),
            revision: "abcdef01".into()
        }
    );
}
