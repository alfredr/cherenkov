//! Inspection progress on stderr; redirected output retains line-based updates.

use cherenkov::model::index::ModelEvent;
use indicatif::{HumanDuration, ProgressBar, ProgressState, ProgressStyle};
use std::{
    fmt::Write,
    io::{self, IsTerminal},
    time::Duration,
};

const SPINNER: &str = "{spinner:.cyan} {wide_msg} [{elapsed_precise}]";
const HEADERS: &str = "{spinner:.cyan} headers {pos}/{len} [{wide_bar:.cyan/dim}] ETA {estimate}";

/// Own the display for one operation, clearing it on success or failure.
/// Cached lookups emit no events, so they never create a spinner.
pub(crate) struct InspectionProgress {
    animated: bool,
    bar: Option<ProgressBar>,
}

impl InspectionProgress {
    pub(crate) fn new() -> Self {
        let stderr = io::stderr();

        Self {
            animated: stderr.is_terminal()
                && anstream::AutoStream::choice(&stderr) != anstream::ColorChoice::Never,
            bar: None,
        }
    }

    pub(crate) fn update(&mut self, event: ModelEvent) {
        if !self.animated {
            plain(event);

            return;
        }

        let bar = self.bar.get_or_insert_with(|| {
            let bar = ProgressBar::new_spinner();

            bar.set_style(style(SPINNER));
            bar.enable_steady_tick(Duration::from_millis(100));

            bar
        });

        match event {
            ModelEvent::Resolving { source } => {
                bar.set_message(format!("Resolving {source}"));
            }
            ModelEvent::Headers {
                completed, total, ..
            } => {
                if completed == 0 {
                    bar.reset();
                    bar.set_style(style(HEADERS));
                }

                bar.set_length(total as u64);
                bar.set_position(completed as u64);
            }
            ModelEvent::ReadingMetadata => {
                bar.set_style(style(SPINNER));
                bar.set_message("Reading model metadata");
                bar.unset_length();
                bar.reset_elapsed();
            }
            _ => {}
        }
    }
}

impl Drop for InspectionProgress {
    fn drop(&mut self) {
        if let Some(bar) = &self.bar {
            bar.finish_and_clear();
        }
    }
}

fn style(template: &str) -> ProgressStyle {
    ProgressStyle::with_template(template)
        .expect("valid inspection progress template")
        .tick_strings(&["-", "\\", "|", "/", " "])
        .progress_chars("=>-")
        .with_key("estimate", |state: &ProgressState, out: &mut dyn Write| {
            if state.pos() == 0 {
                let _ = out.write_str("--");

                return;
            }

            let _ = write!(out, "{}", HumanDuration(state.eta()));
        })
}

fn plain(event: ModelEvent) {
    match event {
        ModelEvent::Resolving { source } => eprintln!("inspect {source}: resolving metadata"),
        ModelEvent::Headers {
            completed,
            total,
            file: Some(file),
        } => eprintln!("inspect: headers {completed}/{total} ({file})"),
        ModelEvent::Headers { total, .. } => eprintln!("inspect: reading {total} shard headers"),
        ModelEvent::ReadingMetadata => eprintln!("inspect: reading model metadata"),
        _ => {}
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/cli_output/progress.rs"]
mod tests;
