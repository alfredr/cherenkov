//! Ratatui inspection progress on stderr, with a plain text fallback.

mod view;

use anyhow::{Context, Result};
use cherenkov::model::index::ModelEvent;
use std::{
    io::{self, IsTerminal, Write},
    sync::mpsc::{self, RecvTimeoutError},
    time::{Duration, Instant},
};
use view::{Progress, StatusLine};

const TICK: Duration = Duration::from_millis(100);

/// Keep terminal rendering on the caller while remote inspection can block.
/// The scoped worker finishes before the display is cleared and results print.
pub(crate) fn inspect_with_progress<T: Send>(
    operation: impl FnOnce(&mut dyn FnMut(ModelEvent)) -> Result<T> + Send,
) -> Result<T> {
    let stderr = io::stderr();

    if !stderr.is_terminal()
        || anstream::AutoStream::choice(&stderr) == anstream::ColorChoice::Never
    {
        return operation(&mut |event| plain(&event));
    }

    let mut line = StatusLine::default();

    drive(operation, |progress| line.draw(progress))
}

fn drive<T: Send>(
    operation: impl FnOnce(&mut dyn FnMut(ModelEvent)) -> Result<T> + Send,
    mut draw: impl FnMut(&Progress) -> io::Result<()>,
) -> Result<T> {
    std::thread::scope(|scope| {
        // Bounded independently of HTTP concurrency. The UI drains events even
        // after a draw failure, so a display error cannot strand the worker.
        let (sender, receiver) = mpsc::sync_channel(16);
        let worker = std::thread::Builder::new()
            .name("model-inspect".into())
            .spawn_scoped(scope, move || {
                operation(&mut |event| {
                    let _ = sender.send(event);
                })
            })
            .context("starting inspection worker")?;

        render_events(receiver, &mut draw);

        worker
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}

fn render_events(
    receiver: mpsc::Receiver<ModelEvent>,
    draw: &mut impl FnMut(&Progress) -> io::Result<()>,
) {
    let mut progress: Option<Progress> = None;
    let mut animated = true;
    let mut next_draw = Instant::now();

    loop {
        match receiver.recv_timeout(next_draw.saturating_duration_since(Instant::now())) {
            Ok(event) => {
                if !animated {
                    plain(&event);

                    continue;
                }

                match &mut progress {
                    Some(progress) => progress.update(event),
                    None => progress = Some(Progress::new(event)),
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }

        if Instant::now() < next_draw {
            continue;
        }

        if animated
            && let Some(progress) = &progress
            && draw(progress).is_err()
        {
            animated = false;

            plain(&progress.event);
        }

        next_draw = Instant::now() + TICK;
    }
}

fn plain(event: &ModelEvent) {
    let message = match event {
        ModelEvent::Resolving { source } => format!("inspect {source}: resolving metadata"),
        ModelEvent::Headers {
            completed,
            total,
            file: Some(file),
        } => {
            format!("inspect: headers {completed}/{total} ({file})")
        }
        ModelEvent::Headers { total, .. } => format!("inspect: reading {total} shard headers"),
        ModelEvent::ReadingMetadata => "inspect: reading model metadata".into(),
        _ => return,
    };

    // Progress is optional; a closed stderr must not change the operation result.
    let _ = writeln!(io::stderr().lock(), "{message}");
}

#[cfg(test)]
#[path = "../../../tests/unit/cli_output/progress.rs"]
mod tests;
