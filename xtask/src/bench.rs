use crate::{
    capture, hardware, metrics, report,
    suite::{self, Case, Configuration, Suite},
    util,
};
use anyhow::{Context, Result, ensure};
use clap::{Args, ValueEnum};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

mod model;
use model::Model;

#[derive(Args)]
pub struct Options {
    /// Indexed model alias, source URI, or prepared directory
    pub model_dir: PathBuf,
    /// Cherenkov index root, forwarded to inspection and inference
    #[arg(long)]
    pub root: Option<PathBuf>,
    #[arg(long, default_value = "benchmarks/suite.json")]
    pub suite: PathBuf,
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub configs: Option<String>,
    #[arg(long)]
    pub cases: Option<String>,
    #[arg(long)]
    pub rounds: Option<usize>,
    #[arg(long)]
    pub case_cap: Vec<String>,
    #[arg(long)]
    pub binary: Option<PathBuf>,
    #[arg(long)]
    pub resume: bool,
    #[arg(long)]
    pub build_stores: bool,
    #[arg(long)]
    pub allow_battery: bool,
    #[arg(long)]
    pub dry_run: bool,
    /// Rebuild the README benchmark section after the suite finishes.
    #[arg(long)]
    pub update_readme: bool,
    /// Free text recorded in the report's provenance (machine, conditions).
    #[arg(long)]
    pub note: Option<String>,
    /// Zip the results directory beside itself when the suite finishes.
    #[arg(long)]
    pub archive: bool,
    /// Preset: `light` runs one round of code, prose and the long prefill on
    /// the stores already built; `heavy` runs the whole suite. Both archive.
    #[arg(long, value_enum)]
    pub mode: Option<Mode>,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Light,
    Heavy,
}

/// Cases the light preset runs when `--cases` is not given.
pub const LIGHT_CASES: &str = "code,prose,prefill-long";

/// Hide runner-supplied locations while preserving prompts and configuration arguments.
pub fn redact_args(args: &mut [Value], config_args: usize) {
    for (arg, replacement) in args.iter_mut().zip(["<binary>", "<model>"]) {
        *arg = json!(replacement);
    }

    let root_flag = 3 + config_args;

    if args.get(root_flag).is_some_and(|arg| arg == "--root")
        && let Some(value) = args.get_mut(root_flag + 1)
    {
        *value = json!("<root>");
    }
}

/// Locations the runner may include in diagnostics, ordered before their parent root.
fn path_replacements(
    binary: &Path,
    model: &Model,
    root: Option<&Path>,
) -> Vec<(String, &'static str)> {
    [
        (Some(binary), "<binary>"),
        (Some(model.directory.as_path()), "<model>"),
        (model.source_directory.as_deref(), "<model>"),
        (root, "<root>"),
    ]
    .into_iter()
    .filter_map(|(path, label)| path.map(|p| (p.to_string_lossy().into_owned(), label)))
    .collect()
}

fn redact_paths(text: &str, replacements: &[(String, &str)]) -> String {
    let mut text = text.to_owned();

    for (path, replacement) in replacements {
        if !path.is_empty() {
            text = text.replace(path, replacement);
        }
    }

    text
}

fn redact_saved_metadata(report: &mut Value, binary: &Path, model: &Model, root: Option<&Path>) {
    let saved_binary = report["provenance"]["binary"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    let saved_model = report["provenance"]["model"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    let mut replacements = path_replacements(binary, model, root);

    replacements.insert(0, (saved_binary, "<binary>"));
    replacements.insert(0, (saved_model, "<model>"));

    let configurations = report["configurations"].clone();

    // Suite contents participate in resume validation. Only redact fields
    // supplied by the runner, leaving prompts and configuration args intact.
    for key in ["runs", "previous_attempts"] {
        if let Some(runs) = report.get_mut(key).and_then(Value::as_array_mut) {
            for run in runs {
                let config_args = configurations
                    .as_array()
                    .and_then(|configs| {
                        configs
                            .iter()
                            .find(|config| config["id"] == run["configuration"])
                    })
                    .and_then(|config| config["args"].as_array())
                    .map_or(0, Vec::len);

                if let Some(args) = run["args"].as_array_mut() {
                    redact_args(args, config_args);
                }

                if let Some(error) = run["error"].as_str() {
                    run["error"] = json!(redact_paths(error, &replacements));
                }
            }
        }
    }

    let provenance = &mut report["provenance"];
    provenance["binary"] = json!("<binary>");
    provenance["model"] = json!("<model>");

    if let Some(platform) = provenance["platform"].as_str() {
        let fields: Vec<_> = platform.split_whitespace().collect();

        // Legacy macOS reports used uname -a: sysname, hostname, release,
        // version (several words), machine. Keep the original machine facts.
        if fields.first() == Some(&"Darwin") && fields.len() > 3 {
            provenance["platform"] = json!(format!(
                "{} {} {}",
                fields[0],
                fields[2],
                fields.last().unwrap()
            ));
        }
    }

    if let Some(settings) = report.get_mut("settings") {
        settings["suite"] = json!("<suite>");
    }
}

fn set_caps(cases: &mut [Case], overrides: &[String]) -> Result<()> {
    for value in overrides {
        let (id, limit) = value
            .split_once('=')
            .context("--case-cap requires ID=TOKENS")?;
        let limit = limit.parse()?;

        ensure!(limit > 0, "token cap must be positive");

        cases
            .iter_mut()
            .find(|c| c.id == id)
            .with_context(|| format!("unknown case: {id}"))?
            .max_tokens = limit;
    }

    Ok(())
}

fn check_stores(model: &Model, configs: &[Configuration], allow_build: bool) -> Result<()> {
    ensure!(
        model.precisions.contains(&4),
        "model must already be packed"
    );

    for config in configs {
        let Some(bits) = config.store_bits else {
            continue;
        };
        let present = model.precisions.contains(&bits);

        ensure!(
            present || allow_build,
            "{bits}-bit store missing; select cached configurations or pass --build-stores"
        );

        if !present {
            eprintln!("Allowing first-use {bits}-bit construction, reported in load time.");
        }
    }

    Ok(())
}

pub fn raise_saved_caps(out: &Path, report: &mut Value, signature: Value) -> Result<()> {
    let revision = json!({"previous_signature":report["signature"],"utc_changed":util::utc()?,"reason":"Raised answer safety caps; prompts, context, binary, and model unchanged."});

    append(report, "suite_revisions", revision);

    let runs = std::mem::take(report["runs"].as_array_mut().context("report runs")?);

    for mut record in runs {
        let args = record["args"].as_array().context("sample args")?;
        let pos = args
            .iter()
            .position(|a| a == "--max-tokens")
            .context("sample token cap")?;
        let old_cap = args[pos + 1]
            .as_str()
            .context("sample cap value")?
            .parse::<u64>()?;
        let cap = signature["cases"]
            .as_array()
            .context("signature cases")?
            .iter()
            .find(|c| c["id"] == record["case"])
            .context("case missing")?["max_tokens"]
            .as_u64()
            .context("token cap")?;

        if record["status"] != "incomplete" || cap <= old_cap {
            append(report, "runs", record);

            continue;
        }

        let archive = format!(
            "attempts/{}-cap-{old_cap}.txt",
            record["id"].as_str().context("sample id")?
        );

        fs::create_dir_all(out.join("attempts"))?;
        fs::rename(
            out.join(record["output"].as_str().context("sample output")?),
            out.join(&archive),
        )?;

        record["output"] = json!(archive);

        append(report, "previous_attempts", record);
    }

    report["cases"] = signature["cases"].clone();
    report["signature"] = signature;

    Ok(())
}

pub fn append(report: &mut Value, key: &str, value: Value) {
    report
        .as_object_mut()
        .unwrap()
        .entry(key.to_owned())
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .unwrap()
        .push(value);
}

fn record_result(
    out: &Path,
    config: &Configuration,
    case: &Case,
    capture: &capture::Captured,
    record: &mut Value,
    allow_build: bool,
    metal_limit_gb: f64,
) -> Result<()> {
    if let Some(cycle) = &capture.cycle {
        record["status"] = json!("cycling");

        anyhow::bail!(
            "stopped after four repeated word blocks: {}",
            cycle["example"]
        );
    }

    ensure!(
        capture.code == 0,
        "engine exited {}: {}",
        capture.code,
        capture
            .stderr
            .chars()
            .rev()
            .take(3000)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    );

    let m = metrics::parse(&capture.stderr)?;
    record["metrics"] = m.clone();

    ensure!(
        m["metal_gb"].as_f64().unwrap() <= metal_limit_gb,
        "reported Metal allocations exceeded this machine's {metal_limit_gb:.1} GB of memory"
    );
    ensure!(
        capture.stable_power(),
        "power source changed during this sample"
    );
    ensure!(
        allow_build || m["store_build_seconds"].is_null(),
        "unexpected store construction; use --build-stores"
    );

    let reason = if m["output_tokens"].as_u64().unwrap() >= case.max_tokens as u64 {
        "length"
    } else {
        "eos"
    };
    record["finish_reason"] = json!(reason);

    if case.stop == "eos" && reason != "eos" {
        record["status"] = json!("incomplete");

        anyhow::bail!("incomplete answer: reached the token safety cap before EOS");
    }

    ensure!(
        case.stop != "length" || reason == "length",
        "fixed-length decode did not reach its token count"
    );

    if case.kind == "svg" {
        record["status"] = json!("invalid_svg");
        let text = fs::read_to_string(out.join(record["output"].as_str().unwrap()))?;
        let (svg, elements) = metrics::extract_svg(&text)?;
        let path = format!("pelicans/{}.svg", config.id);

        fs::write(out.join(&path), format!("{svg}\n"))?;

        record["svg"] = json!(path);
        record["svg_elements"] = json!(elements);
        record["within_element_budget"] = json!(elements <= 120);
    }

    record["status"] = json!("ok");

    Ok(())
}

struct Run<'a> {
    binary: &'a Path,
    model: &'a Model,
    out: &'a Path,
    options: &'a Options,
    max_ctx: usize,
    /// Physical memory in decimal GB; Metal allocations beyond it are an error.
    memory_gb: f64,
}

impl Run<'_> {
    fn sample(&self, config: &Configuration, case: &Case, round: usize) -> Result<Value> {
        let id = format!("r{}-{}-{}", round + 1, case.id, config.id);
        let output = format!("outputs/{id}.txt");
        let mut args = vec![
            self.binary.display().to_string(),
            self.model.reference.clone(),
            case.prompt(),
        ];

        args.extend(config.args.clone());

        if let Some(root) = &self.options.root {
            args.extend(["--root".into(), root.display().to_string()]);
        }

        args.extend([
            "--max-tokens".into(),
            case.max_tokens.to_string(),
            "--max-ctx".into(),
            case.max_ctx.unwrap_or(self.max_ctx).to_string(),
        ]);

        if case.stop == "length" {
            args.push("--no-eos".into());
        }

        eprintln!("[{id}] starting");

        // A previous sample may have published additional expert variants.
        let current = Model::inspect(
            self.binary,
            self.model.reference.as_ref(),
            self.options.root.as_deref(),
        )?;
        let present = config
            .store_bits
            .is_none_or(|bits| current.precisions.contains(&bits));
        let c = capture::run(
            &args,
            &self.out.join(&output),
            self.options.allow_battery,
            None,
        )?;
        let mut record = json!({
            "id": id,
            "configuration": config.id,
            "case": case.id,
            "round": round + 1,
            "args": args,
            "output": output,
            "power_before": c.power_before,
            "power_after": c.power_after,
            "power_samples": c.power_samples,
            "status": "failed",
            "store_present_before": present,
            "wall_seconds": c.wall_seconds,
            "exit_code": c.code,
        });

        redact_args(record["args"].as_array_mut().unwrap(), config.args.len());

        if let Some(cycle) = &c.cycle {
            record["cycle"] = cycle.clone();
        }

        if let Err(error) = record_result(
            self.out,
            config,
            case,
            &c,
            &mut record,
            self.options.build_stores,
            self.memory_gb,
        ) {
            let replacements =
                path_replacements(self.binary, self.model, self.options.root.as_deref());
            record["error"] = json!(redact_paths(&error.to_string(), &replacements));
        }

        eprintln!(
            "[{id}] {}, tg/s {}",
            record["status"], record["metrics"]["tg_s"]
        );

        Ok(record)
    }
    fn new_report(
        &self,
        signature: Value,
        configs: &[Configuration],
        cases: &[Case],
    ) -> Result<Value> {
        let Self {
            binary,
            model,
            options,
            max_ctx,
            ..
        } = self;

        let status = util::output(&["git", "status", "--porcelain"])?;

        eprintln!("Describing hardware and sampling store reads.");

        Ok(json!({
            "version": 1,
            "signature": signature,
            "configurations": configs,
            "cases": cases,
            "runs": [],
            "settings": {
                "suite": "<suite>",
                "mode": options.mode.map(|m| format!("{m:?}").to_lowercase()),
                "max_ctx": max_ctx,
                "case_caps": options.case_cap,
                "build_stores": options.build_stores,
                "binary_override": options.binary.is_some(),
                "pool": "adaptive unless a configuration passes --pool-gb; the server TOML is not read",
                "environment": "CHERENKOV_* variables are removed from every sample",
            },
            "provenance": {
                "commit": util::output(&["git", "rev-parse", "HEAD"])?,
                "dirty": !status.is_empty(),
                "git_status": status,
                "source_sha256": util::source_digest()?,
                "binary_sha256": util::digest(binary)?,
                "binary": "<binary>",
                "model": "<model>",
                "model_metadata_sha256": signature["model_metadata_sha256"],
                "utc_started": util::utc()?,
                "platform": util::output(&["uname", "-srm"])?,
                "hardware": util::output(&["sysctl", "-n", "machdep.cpu.brand_string"])?,
                "memory_bytes": util::output(&["sysctl", "-n", "hw.memsize"])?,
                "rustc": util::output(&["rustc", "--version"])?,
                "power": capture::power(),
                "hardware_detail": hardware::describe(&model.directory),
                "note": options.note,
                "method": "fresh processes; rotated interleaved rounds; separate load/prefill/decode; no check; no prefix cache; complete answers through EOS; SVG phase after timings"
            }
        }))
    }

    fn prepare_report(
        &self,
        signature: Value,
        configs: &[Configuration],
        cases: &[Case],
    ) -> Result<Value> {
        let Self {
            out,
            options,
            binary,
            model,
            ..
        } = self;

        if options.resume {
            let mut report = util::json(&out.join("report.json"))?;

            model.migrate_signature(&mut report["signature"]);

            if let Some(revisions) = report["suite_revisions"].as_array_mut() {
                for revision in revisions {
                    model.migrate_signature(&mut revision["previous_signature"]);
                }
            }

            ensure!(
                report["signature"]["model_id"].is_string(),
                "cannot resume: saved report has no matching model identity; choose a new output directory"
            );

            if report["signature"] != signature {
                ensure!(
                    suite::only_higher_caps(&report["signature"], &signature),
                    "cannot resume: binary, suite, model, or selections changed"
                );
                raise_saved_caps(out, &mut report, signature)?;
            }

            redact_saved_metadata(&mut report, binary, model, options.root.as_deref());

            return Ok(report);
        }

        ensure!(
            !out.exists() || fs::read_dir(out)?.next().is_none(),
            "output directory is not empty; choose another or --resume"
        );
        fs::create_dir_all(out)?;

        self.new_report(signature, configs, cases)
    }
}

/// Inspect once before scheduling; each timed sample refreshes variant availability.
fn resolve_model(options: &Options) -> Result<(PathBuf, Model)> {
    if options.binary.is_none() {
        util::build()?;
    }

    let binary = options
        .binary
        .clone()
        .unwrap_or_else(|| util::root().join("target/release/cherenkov"))
        .canonicalize()?;
    let model = Model::inspect(
        &binary,
        options.model_dir.as_os_str(),
        options.root.as_deref(),
    )?;

    Ok((binary, model))
}

pub fn run(options: Options) -> Result<()> {
    let suite: Suite = serde_json::from_value(util::json(&options.suite)?)?;
    let light = options.mode == Some(Mode::Light);
    let mut configs = suite::select(&suite.configurations, options.configs.as_deref(), |c| &c.id)?;

    let resolved = if !options.dry_run || (light && options.configs.is_none()) {
        Some(resolve_model(&options)?)
    } else {
        None
    };

    // The light preset measures what is already on disk instead of building stores.
    if light && options.configs.is_none() {
        let (_, model) = resolved
            .as_ref()
            .context("light mode requires model inspection")?;

        configs.retain(|c| {
            c.store_bits
                .is_none_or(|bits| model.precisions.contains(&bits))
        });
    }

    let case_selection = options.cases.as_deref().or(light.then_some(LIGHT_CASES));
    let mut cases = suite::select(&suite.cases, case_selection, |c| &c.id)?;

    set_caps(&mut cases, &options.case_cap)?;

    let rounds = options
        .rounds
        .unwrap_or(if light { 1 } else { suite.rounds });

    ensure!(
        rounds > 0 && !configs.is_empty() && !cases.is_empty(),
        "rounds and selections must be nonempty"
    );

    let jobs = suite::schedule(&configs, &cases, rounds);

    if options.dry_run {
        let plan: Vec<_> = jobs
            .iter()
            .map(|&(round, config, case)| {
                json!({
                    "round": round + 1,
                    "config": configs[config].id,
                    "case": cases[case].id,
                    "max_tokens": cases[case].max_tokens,
                })
            })
            .collect();

        println!("{}", serde_json::to_string_pretty(&plan)?);

        return Ok(());
    }

    let (binary, model) = resolved.context("benchmark model was not resolved")?;

    check_stores(&model, &configs, options.build_stores)?;

    let default_name = match options.mode {
        Some(Mode::Light) => format!("{}-light", util::utc()?),
        _ => util::utc()?,
    };
    let out = util::absolute(
        &options
            .output
            .clone()
            .unwrap_or(util::root().join("results").join(default_name)),
    )?;
    let metadata = model.metadata()?;
    let memory_gb = util::output(&["sysctl", "-n", "hw.memsize"])?
        .parse::<f64>()
        .context("hw.memsize")?
        / 1e9;

    let signature = json!({
        "model_metadata_sha256": metadata,
        "binary_sha256": util::digest(&binary)?,
        "suite_sha256": util::digest(&options.suite)?,
        "model": "<model>",
        "model_id": model.reference,
        "configs": configs,
        "cases": cases,
        "rounds": rounds,
        "allow_battery": options.allow_battery,
    });
    let runner = Run {
        binary: &binary,
        model: &model,
        out: &out,
        options: &options,
        max_ctx: suite.max_ctx,
        memory_gb,
    };
    let mut report = runner.prepare_report(signature, &configs, &cases)?;

    fs::create_dir_all(out.join("outputs"))?;
    fs::create_dir_all(out.join("pelicans"))?;
    report::write(&out, &report)?;

    for (r, c, t) in jobs {
        let id = format!("r{}-{}-{}", r + 1, cases[t].id, configs[c].id);

        if report["runs"].as_array().unwrap().iter().any(|v| {
            v["id"] == id
                && matches!(
                    v["status"].as_str(),
                    Some("ok" | "incomplete" | "invalid_svg" | "cycling")
                )
        }) {
            continue;
        }

        let record = runner.sample(&configs[c], &cases[t], r)?;
        let failed = record["status"] == "failed";
        let error = record["error"].clone();

        report["runs"]
            .as_array_mut()
            .unwrap()
            .retain(|v| v["id"] != id);
        append(&mut report, "runs", record);
        report::write(&out, &report)?;
        ensure!(
            !failed,
            "{error}; outputs saved in {}; fix and --resume",
            out.display()
        );
    }

    report["utc_finished"] = json!(util::utc()?);

    report::write(&out, &report)?;

    if options.update_readme {
        crate::readme::update(&out)?;
    }

    println!("{}", out.join("gallery.html").display());

    if options.archive || options.mode.is_some() {
        let zip = hardware::archive(&out)?;

        println!("{}", zip.display());
        eprintln!("Attach the archive to a pull request or issue to share this run.");
    }

    Ok(())
}
