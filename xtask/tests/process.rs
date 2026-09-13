//! Exercise real child processes without allocating GPU memory or reading weights.
use anyhow::Result;
use serde_json::{Value, json};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::PathBuf,
    process::{Command, Output},
    time::{Duration, Instant},
};
use xtask::{capture, util};

struct Fixture {
    directory: tempfile::TempDir,
    binary: PathBuf,
    model: PathBuf,
    suite: PathBuf,
    output: PathBuf,
}

fn telemetry_for_machine(memory_fraction: f64) -> Result<String> {
    let memory_gb = util::output(&["sysctl", "-n", "hw.memsize"])?.parse::<f64>()? / 1e9;
    Ok(include_str!("fixtures/telemetry.txt")
        .replace("20.98", &format!("{:.2}", memory_gb * memory_fraction)))
}

impl Fixture {
    fn new(cap: usize) -> Result<Self> {
        let directory = tempfile::tempdir()?;
        let root = directory.path();
        let binary = root.join("engine");

        fs::write(
            &binary,
            "#!/bin/sh\ndir=$(dirname \"$0\")\necho $$ > \"$dir/pid\"\n[ -z \"${CHERENKOV_FN_FAKE+x}\" ] || exit 99\ncat \"$dir/answer\"\ncat \"$dir/telemetry\" >&2\n",
        )?;
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o700))?;
        fs::write(
            root.join("answer"),
            "A complete answer.\n<svg xmlns=\"http://www.w3.org/2000/svg\"><path/></svg>\n",
        )?;
        fs::write(root.join("telemetry"), telemetry_for_machine(0.5)?)?;

        let model = root.join("model");

        fs::create_dir_all(model.join("packed"))?;

        for name in ["config.json", "tokenizer.json", "packed/manifest.json"] {
            fs::write(model.join(name), "{}")?;
        }

        let suite = root.join("suite.json");

        util::write_json(
            &suite,
            &json!({
                "rounds": 1, "max_ctx": 512,
                "configurations": [{"id": "exact", "label": "4-bit", "args": [], "reproducible_cut": false}],
                "cases": [
                    {"id": "code", "kind": "decode", "prompt": "Explain.", "max_tokens": cap, "stop": "eos"},
                    {"id": "pelican", "kind": "svg", "prompt": "Draw.", "max_tokens": cap, "stop": "eos"}
                ]
            }),
        )?;

        let output = root.join("report");

        Ok(Self {
            directory,
            binary,
            model,
            suite,
            output,
        })
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));

        command
            .arg("bench")
            .arg(&self.model)
            .arg("--binary")
            .arg(&self.binary)
            .arg("--suite")
            .arg(&self.suite)
            .arg("--output")
            .arg(&self.output)
            .arg("--allow-battery")
            .env("CHERENKOV_FN_FAKE", "1");

        command
    }

    fn report(&self) -> Result<Value> {
        util::json(&self.output.join("report.json"))
    }
}

fn successful(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn suite_runs_resumes_and_keeps_svg_passive() -> Result<()> {
    let fixture = Fixture::new(128)?;

    successful(fixture.command().output()?);

    let first = fixture.report()?;

    assert_eq!(first["runs"].as_array().unwrap().len(), 2);
    assert!(
        first["runs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["status"] == "ok")
    );
    assert_eq!(first["runs"][0]["metrics"]["prefill_seconds"], 5.0);
    assert!(fixture.output.join("pelicans/exact.svg").exists());
    successful(fixture.command().arg("--resume").output()?);
    assert_eq!(fixture.report()?["runs"], first["runs"]);

    let html = fs::read_to_string(fixture.output.join("gallery.html"))?;

    assert!(html.contains("<img ") && !html.contains("<svg"));

    Ok(())
}

#[test]
fn shared_archive_redacts_suite_model_and_probe_error_paths() -> Result<()> {
    let fixture = Fixture::new(128)?;

    successful(fixture.command().arg("--archive").output()?);

    let report = fixture.report()?;
    assert_eq!(report["settings"]["suite"], "<suite>");
    assert!(report["provenance"]["hardware_detail"]["store_read"]["error"].is_string());

    let mut archive = zip::ZipArchive::new(fs::File::open(fixture.output.with_extension("zip"))?)?;
    for index in 0..archive.len() {
        let mut contents = String::new();
        std::io::Read::read_to_string(&mut archive.by_index(index)?, &mut contents)?;
        assert!(!contents.contains(fixture.directory.path().to_str().unwrap()));
        assert!(!contents.contains(fixture.directory.path().canonicalize()?.to_str().unwrap()));
    }

    Ok(())
}

#[test]
fn legacy_resume_migrates_paths_and_preserves_completed_samples() -> Result<()> {
    let fixture = Fixture::new(128)?;
    successful(fixture.command().output()?);

    let current = fixture.report()?;
    let mut legacy = current.clone();
    let model = fixture.model.canonicalize()?;
    let binary = fixture.binary.canonicalize()?;
    legacy["signature"]
        .as_object_mut()
        .unwrap()
        .remove("model_path_sha256");
    legacy["signature"]["model"] = json!(model);
    legacy["suite_revisions"] = json!([{"previous_signature": legacy["signature"]}]);
    legacy["provenance"]["model"] = json!(model);
    legacy["provenance"]["binary"] = json!(binary);
    legacy["provenance"]["platform"] = json!(
        "Darwin PRIVATE-PERSON-MACBOOK.local 25.6.0 Darwin Kernel Version 25.6.0: root:xnu/RELEASE_ARM64 arm64"
    );
    legacy["settings"]["suite"] = json!(fixture.suite);
    for run in legacy["runs"].as_array_mut().unwrap() {
        run["args"][0] = json!(binary);
        run["args"][1] = json!(model);
    }
    legacy["previous_attempts"] = legacy["runs"].clone();
    util::write_json(&fixture.output.join("report.json"), &legacy)?;
    fs::remove_file(fixture.directory.path().join("pid"))?;

    successful(
        fixture
            .command()
            .arg("--resume")
            .arg("--archive")
            .output()?,
    );

    let migrated = fixture.report()?;
    assert_eq!(migrated["signature"], current["signature"]);
    assert_eq!(migrated["provenance"]["platform"], "Darwin 25.6.0 arm64");
    assert_eq!(migrated["runs"], current["runs"]);
    assert_eq!(migrated["previous_attempts"], current["runs"]);
    assert_eq!(
        migrated["suite_revisions"][0]["previous_signature"],
        current["signature"]
    );
    assert!(
        !migrated
            .to_string()
            .contains(fixture.directory.path().to_str().unwrap())
    );
    assert!(
        !fixture.directory.path().join("pid").exists(),
        "completed samples reran"
    );

    let mut archive = zip::ZipArchive::new(fs::File::open(fixture.output.with_extension("zip"))?)?;
    for index in 0..archive.len() {
        let mut contents = String::new();
        std::io::Read::read_to_string(&mut archive.by_index(index)?, &mut contents)?;
        assert!(!contents.contains("PRIVATE-PERSON-MACBOOK"));
    }

    Ok(())
}

#[test]
fn repeated_resume_preserves_paths_in_user_supplied_suite_contents() -> Result<()> {
    let fixture = Fixture::new(128)?;
    let model = fixture.model.canonicalize()?;
    let binary = fixture.binary.canonicalize()?;
    let prompt = format!(
        "Write a script to list {} and inspect {}.",
        model.display(),
        binary.display()
    );
    let mut suite = util::json(&fixture.suite)?;
    suite["cases"][0]["prompt"] = json!(prompt);
    suite["configurations"][0]["args"] = json!(["--root", model]);
    util::write_json(&fixture.suite, &suite)?;
    successful(fixture.command().output()?);
    let original = fixture.report()?;
    assert_eq!(original["runs"][0]["args"][2], prompt);
    assert_eq!(original["runs"][0]["args"][4], json!(model));

    for legacy in [false, true] {
        let mut saved = original.clone();
        if legacy {
            saved["signature"]
                .as_object_mut()
                .unwrap()
                .remove("model_path_sha256");
            saved["signature"]["model"] = json!(model);
            saved["provenance"]["binary"] = json!(binary);
            saved["provenance"]["model"] = json!(model);
            for run in saved["runs"].as_array_mut().unwrap() {
                run["args"][0] = json!(binary);
                run["args"][1] = json!(model);
            }
        }
        util::write_json(&fixture.output.join("report.json"), &saved)?;
        for _ in 0..2 {
            successful(fixture.command().arg("--resume").output()?);
            let resumed = fixture.report()?;
            assert_eq!(resumed["signature"], original["signature"]);
            assert_eq!(resumed["runs"], original["runs"]);
            assert_eq!(resumed["cases"], original["cases"]);
            assert_eq!(resumed["configurations"], original["configurations"]);
        }
    }

    Ok(())
}

#[test]
fn resume_rejects_a_different_directory_with_identical_metadata() -> Result<()> {
    let mut fixture = Fixture::new(128)?;
    successful(fixture.command().output()?);

    let original = fixture.report()?;
    let original_model = fixture.model.canonicalize()?;
    let other = fixture.directory.path().join("other-model");
    fs::create_dir_all(other.join("packed"))?;
    for name in ["config.json", "tokenizer.json", "packed/manifest.json"] {
        fs::copy(fixture.model.join(name), other.join(name))?;
    }
    fixture.model = other;

    // Both the current and legacy formats must retain directory validation.
    for legacy in [false, true] {
        let mut saved = original.clone();
        if legacy {
            saved["signature"]
                .as_object_mut()
                .unwrap()
                .remove("model_path_sha256");
            saved["signature"]["model"] = json!(original_model);
        }
        util::write_json(&fixture.output.join("report.json"), &saved)?;
        let result = fixture.command().arg("--resume").output()?;
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("cannot resume"));
        assert_eq!(
            fixture.report()?,
            saved,
            "rejected resume changed the report"
        );
    }

    Ok(())
}

#[test]
fn resume_accepts_an_alias_of_the_same_model_directory() -> Result<()> {
    let mut fixture = Fixture::new(128)?;
    successful(fixture.command().output()?);
    let original = fixture.report()?;
    let alias = fixture.directory.path().join("model-alias");
    std::os::unix::fs::symlink(&fixture.model, &alias)?;
    fixture.model = alias;

    successful(fixture.command().arg("--resume").output()?);
    assert_eq!(fixture.report()?["signature"], original["signature"]);

    Ok(())
}

#[test]
fn light_mode_skips_stores_without_a_completed_manifest() -> Result<()> {
    let fixture = Fixture::new(128)?;
    let mut suite = util::json(&fixture.suite)?;
    suite["configurations"].as_array_mut().unwrap().push(json!({
        "id": "q2", "label": "2-bit", "args": [], "store_bits": 2, "reproducible_cut": false
    }));
    util::write_json(&fixture.suite, &suite)?;
    fs::write(fixture.model.join("packed/experts2.bin"), "unfinished")?;

    successful(
        fixture
            .command()
            .args(["--mode", "light", "--cases", "code"])
            .output()?,
    );
    let report = fixture.report()?;
    assert_eq!(report["configurations"].as_array().unwrap().len(), 1);
    assert_eq!(report["runs"][0]["configuration"], "exact");

    fs::write(fixture.model.join("packed/manifest2.json"), "{}")?;
    let plan = fixture
        .command()
        .args(["--mode", "light", "--cases", "code", "--dry-run"])
        .output()?;
    assert!(plan.status.success());
    let plan: Value = serde_json::from_slice(&plan.stdout)?;
    assert!(
        plan.as_array()
            .unwrap()
            .iter()
            .any(|job| job["config"] == "q2")
    );

    Ok(())
}

#[test]
fn capped_answers_are_retained_but_excluded() -> Result<()> {
    let fixture = Fixture::new(64)?;

    successful(fixture.command().output()?);

    let report = fixture.report()?;

    assert!(
        report["runs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["status"] == "incomplete")
    );
    assert!(xtask::report::rows(&report)?.is_empty());
    assert!(!fixture.output.join("pelicans/exact.svg").exists());

    Ok(())
}

#[test]
fn failed_memory_check_stops_then_resume_retries() -> Result<()> {
    let fixture = Fixture::new(128)?;
    let telemetry = fs::read_to_string(fixture.directory.path().join("telemetry"))?;

    fs::write(
        fixture.directory.path().join("telemetry"),
        telemetry_for_machine(2.0)?,
    )?;
    assert!(!fixture.command().output()?.status.success());

    let failed = fixture.report()?;

    assert_eq!(failed["runs"].as_array().unwrap().len(), 1);
    assert_eq!(failed["runs"][0]["status"], "failed");
    assert!(
        failed["runs"][0]["error"]
            .as_str()
            .unwrap()
            .contains("GB of memory")
    );
    fs::write(fixture.directory.path().join("telemetry"), telemetry)?;
    successful(fixture.command().arg("--resume").output()?);
    assert!(
        fixture.report()?["runs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|r| r["status"] == "ok")
    );

    Ok(())
}

#[test]
fn timeout_reaps_the_engine() -> Result<()> {
    let fixture = Fixture::new(128)?;
    let args = vec!["/bin/sleep".into(), "60".into()];
    let start = Instant::now();
    let result = capture::run(
        &args,
        &fixture.directory.path().join("out"),
        true,
        Some(Duration::from_millis(100)),
    )?;

    assert!(result.timed_out && result.code != 0);
    assert!(start.elapsed() < Duration::from_secs(5));

    Ok(())
}

#[test]
fn interrupt_reaps_the_engine_and_leaves_a_resumable_report() -> Result<()> {
    let fixture = Fixture::new(128)?;

    fs::write(
        &fixture.binary,
        "#!/bin/sh\ndir=$(dirname \"$0\")\necho $$ > \"$dir/pid\"\nexec /bin/sleep 60\n",
    )?;

    let mut parent = capture::ChildGuard(
        fixture
            .command()
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?,
    );
    let pid_path = fixture.directory.path().join("pid");
    let start = Instant::now();

    while !pid_path.exists() {
        assert!(start.elapsed() < Duration::from_secs(10));
        assert!(parent.0.try_wait()?.is_none());
        std::thread::sleep(Duration::from_millis(20));
    }

    let pid: libc::pid_t = fs::read_to_string(pid_path)?.trim().parse()?;

    // Signal only the runner: its child must be explicitly stopped during unwind.
    assert_eq!(
        unsafe { libc::kill(parent.0.id() as libc::pid_t, libc::SIGTERM) },
        0
    );

    let status = loop {
        if let Some(status) = parent.0.try_wait()? {
            break status;
        }

        assert!(start.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(20));
    };

    assert!(!status.success());
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
    assert_eq!(fixture.report()?["runs"], json!([]));
    assert!(fixture.output.join("outputs/r1-code-exact.txt").exists());

    Ok(())
}
