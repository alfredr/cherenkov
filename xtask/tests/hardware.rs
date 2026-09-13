use anyhow::Result;
use serde_json::json;
use std::io::Write;
use std::path::Path;
use xtask::{bench, hardware, report};

fn profiler_fixture() -> serde_json::Value {
    json!({
        "SPHardwareDataType": [{
            "machine_name": "MacBook Air", "machine_model": "Mac16,13", "chip_type": "Apple M4",
            "number_processors": "proc 10:4:6:0", "physical_memory": "32 GB",
            "serial_number": "SECRET", "platform_UUID": "SECRET", "provisioning_UDID": "SECRET"
        }],
        "SPDisplaysDataType": [{
            "sppci_model": "Apple M4", "sppci_cores": "10", "spdisplays_mtlgpufamilysupport": "spdisplays_metal4"
        }],
        "SPNVMeDataType": [{
            "_name": "Apple SSD Controller",
            "_items": [{
                "device_model": "APPLE SSD AP2048Z", "size": "2 TB", "size_in_bytes": 2001111162880u64,
                "bsd_name": "disk0", "device_serial": "SECRET"
            }]
        }]
    })
}

#[test]
fn profile_keeps_hardware_class_and_drops_identifiers() {
    let profile = hardware::profile_fields(&profiler_fixture());

    assert_eq!(profile["chip"], "Apple M4");
    assert_eq!(profile["gpu_cores"], "10");
    assert_eq!(profile["memory"], "32 GB");
    assert_eq!(profile["nvme"][0]["model"], "APPLE SSD AP2048Z");
    assert_eq!(profile["nvme"][0]["size_bytes"], 2001111162880u64);
    assert!(!profile.to_string().contains("SECRET"));
}

#[test]
fn portable_facts_come_from_libc() -> Result<()> {
    let kernel = hardware::kernel().unwrap();

    assert_eq!(kernel.split(' ').count(), 3, "{kernel}");
    assert!(hardware::physical_memory().unwrap() > 1 << 30);

    let dir = tempfile::tempdir()?;
    let volume = hardware::volume(dir.path())?;

    assert!(volume["total_bytes"].as_u64().unwrap() >= volume["free_bytes"].as_u64().unwrap());
    assert!(volume["free_bytes"].as_u64().unwrap() > 0);
    assert!(hardware::volume(Path::new("/definitely/not/here")).is_err());

    Ok(())
}

#[test]
fn read_rate_samples_the_requested_bytes_across_threads() -> Result<()> {
    let mut file = tempfile::NamedTempFile::new()?;

    file.write_all(&vec![7u8; 4 * 1024 * 1024])?;

    let sample = hardware::read_rate(file.path(), 3 * 1024 * 1024, 3)?;

    assert_eq!(sample["bytes"], 3 * 1024 * 1024);
    assert_eq!(sample["threads"], 3);
    assert!(sample["gbps"].as_f64().unwrap() > 0.0);

    // A request larger than the file reads the whole file once.
    let whole = hardware::read_rate(file.path(), 1 << 40, 2)?;

    assert_eq!(whole["bytes"], 4 * 1024 * 1024);
    assert!(hardware::read_rate(file.path(), 1, 0).is_err());

    Ok(())
}

#[test]
fn hardware_line_summarizes_a_report_and_tolerates_older_ones() {
    let provenance = json!({
        "hardware_detail": {
            "profile": hardware::profile_fields(&profiler_fixture()),
            "store_read": {"gbps": 3.04},
            "memory_bytes": 34359738368u64,
            "kernel": "Darwin 25.6.0 arm64",
            "os_version": "26.6.2",
            "model_volume": {}
        },
        "note": "AC power, idle machine"
    });

    assert_eq!(
        report::hardware_line(&provenance).unwrap(),
        "Hardware: Apple M4 with a 10-core GPU, 32 GiB memory, APPLE SSD AP2048Z 2 TB, expert store reads at 3.0 GB/s, macOS 26.6.2. Note: AC power, idle machine"
    );
    assert!(report::hardware_line(&json!({"commit": "abc"})).is_none());
}

#[test]
fn shared_reports_carry_no_home_directory_paths() {
    let model = Path::new("/Users/someone/.cache/models/flash");
    let binary = Path::new("/Users/someone/src/cherenkov/target/release/cherenkov");
    let args: Vec<String> = [
        binary.to_str().unwrap(),
        model.to_str().unwrap(),
        "Write a function.",
        "--experts",
        "4",
    ]
    .map(String::from)
    .to_vec();
    let redacted = bench::redact_args(&args, model, binary);

    assert_eq!(redacted[0], "<binary>");
    assert_eq!(redacted[1], "<model>");
    assert_eq!(&redacted[2..], &args[2..]);
    assert!(!redacted.join(" ").contains("someone"));
}

#[test]
fn archive_zips_the_results_directory_beside_itself() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let out = dir.path().join("run-1");

    std::fs::create_dir_all(out.join("outputs"))?;
    std::fs::write(out.join("report.json"), "{}")?;
    std::fs::write(out.join("outputs/a.txt"), "answer")?;

    let zip = hardware::archive(&out)?;

    assert_eq!(zip, dir.path().join("run-1.zip"));

    let mut archive = zip::ZipArchive::new(std::fs::File::open(&zip)?)?;
    let names: Vec<String> = archive.file_names().map(str::to_owned).collect();

    assert!(names.contains(&"run-1/report.json".to_owned()), "{names:?}");
    assert!(
        names.contains(&"run-1/outputs/a.txt".to_owned()),
        "{names:?}"
    );

    let mut answer = String::new();

    std::io::Read::read_to_string(&mut archive.by_name("run-1/outputs/a.txt")?, &mut answer)?;
    assert_eq!(answer, "answer");

    // Archiving again replaces the file rather than appending to it.
    hardware::archive(&out)?;
    assert_eq!(
        zip::ZipArchive::new(std::fs::File::open(&zip)?)?.len(),
        names.len()
    );

    Ok(())
}

#[test]
fn describe_reports_this_machine_when_a_model_is_available() {
    let Some(model) = std::env::var_os("CHERENKOV_MODEL_DIR") else {
        return;
    };
    let detail = hardware::describe(Path::new(&model));
    let text = detail.to_string();

    assert!(detail["kernel"].is_string(), "{text}");
    assert!(detail["memory_bytes"].as_u64().unwrap() > 1 << 30, "{text}");
    assert!(
        detail["model_volume"]["total_bytes"].as_u64().unwrap() > 0,
        "{text}"
    );

    if cfg!(target_os = "macos") {
        assert!(detail["profile"]["chip"].is_string(), "{text}");
        assert!(detail["os_version"].is_string(), "{text}");
    }

    assert!(
        detail["store_read"]["gbps"].as_f64().unwrap_or(0.0) > 0.0,
        "{text}"
    );
    assert_eq!(detail["store_read"]["file"], "experts.bin");
    assert!(!text.contains("serial"), "{text}");
    assert!(!text.contains("/Users/"), "{text}");
}

#[test]
fn failed_probes_do_not_expose_the_model_path() {
    let model = Path::new("/Users/private-person/definitely-missing-model");
    let detail = hardware::describe(model);

    assert!(detail["store_read"]["error"].is_string());
    assert!(detail["model_volume"]["error"].is_string());
    assert!(!detail.to_string().contains("private-person"));
}
