//! Markdown and file:// HTML reports use the same filtered measurements.
use crate::util;
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::{fmt::Write as _, fs, path::Path};

pub(crate) mod markdown;

pub fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);

    let n = values.len();

    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

pub fn text(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

pub fn rows(report: &Value) -> Result<Vec<Value>> {
    let mut rows = Vec::new();

    for case in report["cases"].as_array().context("report cases")? {
        for config in report["configurations"]
            .as_array()
            .context("report configurations")?
        {
            let matching: Vec<_> = report["runs"]
                .as_array()
                .context("report runs")?
                .iter()
                .filter(|r| {
                    r["case"] == case["id"]
                        && r["configuration"] == config["id"]
                        && r["status"] == "ok"
                })
                .map(|r| &r["metrics"])
                .collect();

            if matching.is_empty() {
                continue;
            }

            let memory = matching
                .iter()
                .filter_map(|r| r["metal_gb"].as_f64())
                .fold(0.0, f64::max);
            let mut row = json!({
                "case": case["id"],
                "kind": case["kind"].as_str().unwrap_or("decode"),
                "configuration": config["id"],
                "label": config["label"],
                "runs": matching.len(),
                "metal_gb": memory,
            });

            for key in [
                "prompt_tokens",
                "prefill_seconds",
                "output_tokens",
                "decode_seconds",
                "load_seconds",
                "pp_s",
                "tg_s",
            ] {
                let values = matching
                    .iter()
                    .map(|r| r[key].as_f64().with_context(|| format!("missing {key}")))
                    .collect::<Result<Vec<_>>>()?;
                row[key] = json!(median(values));
            }

            rows.push(row);
        }
    }

    Ok(rows)
}

fn number(row: &Value, key: &str) -> f64 {
    row[key].as_f64().unwrap_or(0.0)
}

fn case_label(id: &str) -> &str {
    match id {
        "code" => "Linked-list code",
        "code-lru" => "LRU cache",
        "debug-bisect" => "Binary-search debugging",
        "prose" => "Hash-table explanation",
        "reasoning" => "Scheduling arithmetic",
        "structured" => "JSON extraction",
        "prefill-long" => "Long document",
        "pelican" => "Pelican SVG",
        _ => id,
    }
}

fn config_label(config: &Value) -> &str {
    match text(&config["id"]) {
        "exact-4bit" => "4-bit",
        "misses-2bit" => "4/2-bit + cut",
        "all-3bit" => "3-bit",
        "all-2bit" => "2-bit",
        _ => text(&config["label"]),
    }
}

/// One line of machine facts for reports that carry `hardware_detail`.
pub fn hardware_line(provenance: &Value) -> Option<String> {
    let detail = provenance.get("hardware_detail")?;
    let profile = &detail["profile"];
    let mut parts = Vec::new();

    if !profile["chip"].is_null() {
        parts.push(match profile["gpu_cores"].as_str() {
            Some(cores) => format!("{} with a {cores}-core GPU", text(&profile["chip"])),
            None => text(&profile["chip"]).to_owned(),
        });
    }

    if let Some(bytes) = detail["memory_bytes"].as_u64() {
        parts.push(format!("{} GiB memory", bytes >> 30));
    } else if !profile["memory"].is_null() {
        parts.push(format!("{} memory", text(&profile["memory"])));
    }

    if let Some(drive) = profile["nvme"].as_array().and_then(|d| d.first()) {
        parts.push(format!(
            "{} {}",
            text(&drive["model"]),
            text(&drive["size"])
        ));
    }

    if let Some(gbps) = detail["store_read"]["gbps"].as_f64() {
        parts.push(format!("expert store reads at {gbps:.1} GB/s"));
    }

    if let Some(version) = detail["os_version"].as_str() {
        parts.push(
            if detail["kernel"]
                .as_str()
                .is_some_and(|k| k.starts_with("Darwin"))
            {
                format!("macOS {version}")
            } else {
                version.to_owned()
            },
        );
    } else if let Some(kernel) = detail["kernel"].as_str() {
        parts.push(kernel.to_owned());
    }

    let note = provenance["note"]
        .as_str()
        .map(|n| format!(" Note: {n}"))
        .unwrap_or_default();

    (!parts.is_empty()).then(|| format!("Hardware: {}.{note}", parts.join(", ")))
}

pub fn write(out: &Path, report: &Value) -> Result<()> {
    util::write_json(&out.join("report.json"), report)?;

    regenerate(out, report)
}

pub fn regenerate(out: &Path, report: &Value) -> Result<()> {
    let rows = rows(report)?;
    let mut summary = format!(
        "# Cherenkov benchmark\n\nSource commit: `{}`\n\n",
        text(&report["provenance"]["commit"])
    );

    if let Some(line) = hardware_line(&report["provenance"]) {
        summary.push_str(&line);
        summary.push_str("\n\n");
    }

    summary.push_str("| Case | Configuration | Runs | Output tokens | Decode s | Load s | pp/s | tg/s | Metal GB |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");

    for row in &rows {
        writeln!(
            summary,
            "| {} | {} | {} | {:.0} | {:.2} | {:.2} | {:.1} | {:.2} | {:.2} |",
            text(&row["case"]),
            text(&row["label"]),
            row["runs"],
            number(row, "output_tokens"),
            number(row, "decode_seconds"),
            number(row, "load_seconds"),
            number(row, "pp_s"),
            number(row, "tg_s"),
            number(row, "metal_gb")
        )?;
    }

    if out.join("README.md").is_file() {
        summary.push_str("\n[Run observations](README.md).\n");
    }

    append_failures(&mut summary, report)?;

    let pelicans = markdown::drawings(report, Path::new(""))?;

    if !pelicans.is_empty() {
        summary
            .push_str("\n## Pelicans\n\nThese are unedited model outputs from the benchmark.\n\n");
        summary.push_str(pelicans.trim_end());
        summary.push('\n');
    }

    summary.push_str(&markdown::answers(report)?);
    fs::write(out.join("summary.md"), summary)?;
    fs::write(out.join("gallery.html"), gallery(report, &rows)?)?;

    Ok(())
}

fn append_failures(summary: &mut String, report: &Value) -> Result<()> {
    let bad: Vec<_> = report["runs"]
        .as_array()
        .context("report runs")?
        .iter()
        .filter(|r| r["status"] != "ok")
        .collect();

    if !bad.is_empty() {
        summary.push_str("\n## Incomplete or invalid runs\n\n");

        for r in bad {
            writeln!(
                summary,
                "- {}: {}",
                text(&r["id"]),
                r["error"].as_str().unwrap_or(text(&r["status"]))
            )?;
        }
    }

    Ok(())
}

fn table(report: &Value, rows: &[Value], kind: &str, key: &str, caption: &str) -> Result<String> {
    let configs = report["configurations"]
        .as_array()
        .context("configurations")?;
    let mut table = format!(
        "<div class=table-scroll tabindex=0><table><caption>{caption}</caption><thead><tr><th scope=col>Workload</th>"
    );

    for c in configs {
        write!(
            table,
            "<th scope=col title=\"{}\">{}</th>",
            escape(text(&c["label"])),
            escape(config_label(c))
        )?;
    }

    table.push_str("</tr></thead><tbody>");

    for case in report["cases"].as_array().context("cases")? {
        if case["kind"].as_str().unwrap_or("decode") != kind {
            continue;
        }

        write!(
            table,
            "<tr><th scope=row>{}</th>",
            escape(case_label(text(&case["id"])))
        )?;

        for config in configs {
            match rows
                .iter()
                .find(|r| r["case"] == case["id"] && r["configuration"] == config["id"])
            {
                Some(row) => write!(table, "<td>{:.2}</td>", number(row, key))?,
                None => table.push_str("<td>Pending</td>"),
            }
        }

        table.push_str("</tr>");
    }

    table.push_str("</tbody></table></div>");

    Ok(table)
}

fn details(rows: &[Value]) -> Result<String> {
    let fields = [
        ("runs", "Runs"),
        ("prompt_tokens", "Prompt tokens"),
        ("prefill_seconds", "Prefill s"),
        ("pp_s", "pp/s"),
        ("output_tokens", "Output tokens"),
        ("decode_seconds", "Decode s"),
        ("tg_s", "tg/s"),
        ("load_seconds", "Load s"),
        ("metal_gb", "Metal GB"),
    ];
    let mut s = String::from(
        "<details><summary>Full phase timings and answer lengths</summary><div class=table-scroll tabindex=0><table class=details-table><thead><tr><th>Workload</th><th>Configuration</th>",
    );

    for (_, label) in fields {
        write!(s, "<th>{label}</th>")?;
    }

    s.push_str("</tr></thead><tbody>");

    for row in rows {
        write!(
            s,
            "<tr><th scope=row>{}</th><td class=label>{}</td>",
            escape(case_label(text(&row["case"]))),
            escape(text(&row["label"]))
        )?;

        for (key, _) in fields {
            let decimals = match key {
                "runs" | "prompt_tokens" | "output_tokens" => 0,
                _ => 2,
            };

            write!(s, "<td>{:.*}</td>", decimals, number(row, key))?;
        }

        s.push_str("</tr>");
    }

    s.push_str("</tbody></table></div></details>");

    Ok(s)
}

fn cards(report: &Value) -> Result<String> {
    let mut cards = String::new();

    for case in report["cases"].as_array().context("cases")? {
        if case["kind"] != "svg" {
            continue;
        }

        for config in report["configurations"]
            .as_array()
            .context("configurations")?
        {
            write!(
                cards,
                "<article><h3>{}</h3>",
                escape(text(&config["label"]))
            )?;

            let run = report["runs"]
                .as_array()
                .context("runs")?
                .iter()
                .rev()
                .find(|r| r["case"] == case["id"] && r["configuration"] == config["id"]);

            if let Some(run) = run {
                card_body(&mut cards, run)?;
            } else {
                cards.push_str("<p>The drawing is queued after the timing workloads.</p>");
            }

            cards.push_str("</article>");
        }
    }

    if cards.is_empty() {
        cards.push_str("<p>No drawing workloads selected.</p>");
    }

    Ok(cards)
}

fn card_body(card: &mut String, run: &Value) -> Result<()> {
    if let Some(svg) = run["svg"].as_str() {
        // Keep model SVGs passive; never insert generated markup into the page.
        write!(
            card,
            "<img src=\"{}\" alt=\"Generated pelican riding a bicycle\"><p>{} elements; {} tokens</p><p>{:.2} tokens/s; {:.2} s generation</p>",
            escape(svg),
            run["svg_elements"],
            run["metrics"]["output_tokens"],
            number(&run["metrics"], "tg_s"),
            number(&run["metrics"], "decode_seconds")
        )?;
    } else {
        write!(
            card,
            "<p>The run produced no completed SVG: {}</p>",
            escape(run["error"].as_str().unwrap_or(text(&run["status"])))
        )?;
    }

    if let Some(output) = run["output"].as_str() {
        write!(card, "<a href=\"{}\">Full output</a>", escape(output))?;
    }

    Ok(())
}

fn progress(report: &Value) -> Result<String> {
    let runs = report["runs"].as_array().context("runs")?;
    let configs = report["configurations"]
        .as_array()
        .context("configurations")?;
    let rounds = report["signature"]["rounds"].as_u64().unwrap_or(1);
    let samples_per_config: u64 = report["cases"]
        .as_array()
        .context("cases")?
        .iter()
        .map(|case| {
            if case["kind"] == "svg" {
                1
            } else {
                rounds.min(case["rounds"].as_u64().unwrap_or(rounds))
            }
        })
        .sum();
    let planned = configs.len() as u64 * samples_per_config;
    let valid = runs.iter().filter(|r| r["status"] == "ok").count();
    let drawings = runs.iter().filter(|r| r["svg"].is_string()).count();

    Ok(format!(
        "The run has {valid} of {planned} valid samples and {drawings} drawings."
    ))
}

fn gallery(report: &Value, rows: &[Value]) -> Result<String> {
    let mut page =
        include_str!("../templates/gallery.html").replace("{{progress}}", &progress(report)?);
    let tables = table(report, rows, "decode", "tg_s", "Generation: tokens/s")?
        + &table(
            report,
            rows,
            "prefill",
            "pp_s",
            "Prompt processing: tokens/s",
        )?
        + &details(rows)?;
    page = page
        .replace("{{timings}}", &tables)
        .replace("{{cards}}", &cards(report)?);
    let refresh = if report.get("utc_finished").is_none() {
        "<meta http-equiv=\"refresh\" content=\"30\">"
    } else {
        ""
    };

    Ok(page.replace("{{refresh}}", refresh))
}
