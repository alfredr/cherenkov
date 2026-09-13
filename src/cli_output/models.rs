use super::{Report, Section, TableData, terminal};
use anyhow::Result;
use cherenkov::model::{
    Architecture, ModelDescription,
    index::{GcReport, ModelDetails, ModelSummary, Removal, Source},
};
use serde::Serialize;
use std::io::IsTerminal;

pub(crate) fn list(models: &[ModelSummary], json: bool) -> Result<()> {
    let rows = models
        .iter()
        .map(|m| {
            vec![
                m.id.clone(),
                m.reference.clone(),
                architecture(&m.architecture),
                source(&m.source),
                m.source_local.to_string(),
                bits(&m.precisions),
                m.owned_bytes.to_string(),
            ]
        })
        .collect();
    let mut report = Report::default();

    report.sections.push(Section::Table {
        title: "Models".into(),
        data: TableData {
            headers: [
                "ID",
                "Reference",
                "Architecture",
                "Source",
                "Local source",
                "Prepared bits",
                "Owned bytes",
            ]
            .map(str::to_owned)
            .to_vec(),
            rows,
        },
    });

    print(&models, &report, json)
}

pub(crate) fn show(model: &ModelDetails, json: bool) -> Result<()> {
    let m = &model.summary;
    let mut report = Report::default();

    report.sections.push(Section::Fields {
        title: m.reference.clone(),
        values: vec![
            ("ID".into(), m.id.clone()),
            ("Architecture".into(), architecture(&m.architecture)),
            ("Source".into(), source(&m.source)),
            (
                "Source available locally".into(),
                m.source_local.to_string(),
            ),
            ("Prepared bits".into(), bits(&m.precisions)),
            ("Owned bytes".into(), m.owned_bytes.to_string()),
            (
                "External artifact bytes".into(),
                m.external_bytes.to_string(),
            ),
        ],
    });

    for artifact in &model.artifacts {
        report.note(format!(
            "{}: {} ({}, {} bytes)",
            artifact.id,
            artifact.path.display(),
            if artifact.owned { "owned" } else { "external" },
            artifact.bytes
        ));
    }

    print(model, &report, json)
}

pub(crate) fn removed(result: &Removal, json: bool) -> Result<()> {
    let mut report = Report::default();

    report.note(format!("{}: released {} artifact reference(s). Run `cherenkov model gc --dry-run` to inspect cleanup.", result.id, result.released_artifacts.len()));

    print(result, &report, json)
}

pub(crate) fn gc(result: &GcReport, json: bool) -> Result<()> {
    let mut report = Report::default();

    report.note(format!(
        "{} {} artifact(s), {} referenced file bytes; {} in use.",
        if result.dry_run {
            "Would remove"
        } else {
            "Removed"
        },
        result.artifacts.len(),
        result.candidate_bytes,
        result.leased.len()
    ));

    for id in &result.artifacts {
        report.note(id.clone());
    }

    print(result, &report, json)
}

/// Show index provenance and tensor metadata together, in either output format.
pub(crate) fn inspect_indexed(
    details: &ModelDetails,
    description: &ModelDescription,
    json: bool,
) -> Result<()> {
    let precisions = &details.summary.precisions;
    let result = serde_json::json!({"model": description, "preparation": description.preparation(),
        "index": details, "prepared_precisions": precisions, "tensor_view": if precisions.is_empty() { "source" } else { "q4_base" }});
    let mut report = Report::default();

    report.sections.push(Section::Fields {
        title: details.summary.reference.clone(),
        values: vec![
            ("ID".into(), details.summary.id.clone()),
            ("Source".into(), source(&details.summary.source)),
            (
                "Owned bytes".into(),
                details.summary.owned_bytes.to_string(),
            ),
            (
                "External artifact bytes".into(),
                details.summary.external_bytes.to_string(),
            ),
            (
                "Architecture".into(),
                architecture(&description.architecture),
            ),
            ("Tensors".into(), description.tensors.len().to_string()),
            ("Prepared bits".into(), bits(precisions)),
            (
                "Preparation".into(),
                serde_json::to_string(&description.preparation())?,
            ),
        ],
    });

    if !precisions.is_empty() {
        report.note("Tensor details describe the Q4 base. Prepared bits lists the available expert variants.".into());
    }

    print(&result, &report, json)
}

/// Print store registrations with the shared terminal renderer or as raw JSON.
pub(crate) fn stores(stores: &[cherenkov::model::index::DiskStore], json: bool) -> Result<()> {
    use cherenkov::model::index::DiskLayout;

    let rows = stores
        .iter()
        .map(|store| {
            vec![
                store.name.clone(),
                match store.layout {
                    DiskLayout::Directory => "directory",
                    DiskLayout::HfCache => "hf-cache",
                }
                .to_owned(),
                store.path.display().to_string(),
                store.enabled.to_string(),
            ]
        })
        .collect();
    let mut report = Report::default();

    report.sections.push(Section::Table {
        title: "Stores".into(),
        data: TableData {
            headers: ["Name", "Layout", "Root", "Enabled"]
                .map(str::to_owned)
                .to_vec(),
            rows,
        },
    });

    print(&stores, &report, json)
}

fn print(value: &impl Serialize, report: &Report, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(value)?);

        return Ok(());
    }

    if std::io::stdout().is_terminal() {
        return terminal::print(report);
    }

    println!("{}", report.markdown());

    Ok(())
}

fn bits(bits: &[u32]) -> String {
    if bits.is_empty() {
        return "none".into();
    }

    bits.iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn architecture(architecture: &Architecture) -> String {
    match architecture {
        Architecture::Qwen4Exp => "qwen4_exp".into(),
        Architecture::Opaque { name } => name.clone(),
    }
}

fn source(source: &Source) -> String {
    match source {
        Source::Local { path, .. } => path.display().to_string(),
        Source::HuggingFace { repo, revision, .. } => format!("hf://{repo}@{revision}"),
    }
}
