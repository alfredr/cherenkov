//! A Ratatui status row using the same theme as reports and the dashboard.

use super::ModelEvent;
use crate::cli_output::terminal::theme::Theme;
use ratatui::{
    backend::IntoCrossterm,
    buffer::Buffer,
    crossterm::{
        cursor::MoveToColumn,
        queue,
        style::{Attribute, Print, SetAttribute, SetStyle},
        terminal::{self, Clear, ClearType},
    },
    layout::{Constraint, Layout, Rect},
    text::Span,
    widgets::{LineGauge, Paragraph, Widget},
};
use std::{
    io::{self, Write},
    time::{Duration, Instant},
};

pub(super) struct Progress {
    pub(super) event: ModelEvent,
    started: Instant,
}

impl Progress {
    pub(super) fn new(event: ModelEvent) -> Self {
        Self {
            event,
            started: Instant::now(),
        }
    }

    pub(super) fn update(&mut self, event: ModelEvent) {
        if !matches!(event, ModelEvent::Headers { completed: 1.., .. }) {
            self.started = Instant::now();
        }

        self.event = event;
    }

    pub(super) fn render(&self, width: u16, elapsed: Duration) -> Buffer {
        let area = Rect::new(0, 0, width, 1);
        let mut buffer = Buffer::empty(area);
        let theme = Theme::new(true);
        let [spinner, body] =
            Layout::horizontal([Constraint::Length(2), Constraint::Min(0)]).areas(area);
        let glyph = ["-", "\\", "|", "/"][(elapsed.as_millis() / 100 % 4) as usize];

        Paragraph::new(Span::styled(glyph, theme.heading)).render(spinner, &mut buffer);

        match &self.event {
            ModelEvent::Headers {
                completed, total, ..
            } => {
                let remaining = eta(*completed, *total, elapsed)
                    .map_or_else(|| "--".into(), |seconds| format!("{seconds}s"));
                let ratio = (*completed as f64 / (*total).max(1) as f64).clamp(0.0, 1.0);

                LineGauge::default()
                    .label(format!("headers {completed}/{total} ETA {remaining}"))
                    .filled_style(theme.heading)
                    .unfilled_style(theme.rule)
                    .ratio(ratio)
                    .render(body, &mut buffer);
            }
            ModelEvent::Resolving { source } => {
                Paragraph::new(format!("Resolving {source} ({}s)", elapsed.as_secs()))
                    .render(body, &mut buffer);
            }
            ModelEvent::ReadingMetadata => {
                Paragraph::new(format!("Reading model metadata ({}s)", elapsed.as_secs()))
                    .render(body, &mut buffer);
            }
            _ => {}
        }

        buffer
    }
}

/// Estimate this phase from completed headers, never from scheduled requests.
fn eta(completed: usize, total: usize, elapsed: Duration) -> Option<u64> {
    if completed == 0 {
        return None;
    }

    Some(
        (elapsed.as_secs_f64() * total.saturating_sub(completed) as f64 / completed as f64).ceil()
            as u64,
    )
}

/// Write only the current row. Standard inline initialization queries the cursor
/// through stdout, which would corrupt piped JSON; this display needs no query.
#[derive(Default)]
pub(super) struct StatusLine {
    drawn: bool,
}

impl StatusLine {
    pub(super) fn draw(&mut self, progress: &Progress) -> io::Result<()> {
        let result = self.draw_row(progress);

        if result.is_err() {
            self.clear();
        }

        result
    }

    fn draw_row(&mut self, progress: &Progress) -> io::Result<()> {
        let (width, _) = terminal::size()?;
        // Leave the last column unused to avoid wrapping at the terminal edge.
        let buffer = progress.render(width.saturating_sub(1), progress.started.elapsed());

        self.drawn = true;

        write_row(&mut io::stderr().lock(), &buffer)
    }

    fn clear(&mut self) {
        if !self.drawn {
            return;
        }

        let _ = clear_row(&mut io::stderr().lock());

        self.drawn = false;
    }
}

impl Drop for StatusLine {
    fn drop(&mut self) {
        self.clear();
    }
}

fn write_row(out: &mut impl Write, buffer: &Buffer) -> io::Result<()> {
    queue!(out, MoveToColumn(0), Clear(ClearType::CurrentLine))?;

    // Buffer::diff skips continuation cells of wide glyphs.
    for (x, _, cell) in Buffer::empty(buffer.area).diff(buffer) {
        queue!(
            out,
            MoveToColumn(x),
            SetAttribute(Attribute::Reset),
            SetStyle(cell.style().into_crossterm()),
            Print(cell.symbol())
        )?;
    }

    queue!(out, SetAttribute(Attribute::Reset), MoveToColumn(0))?;

    out.flush()
}

fn clear_row(out: &mut impl Write) -> io::Result<()> {
    queue!(
        out,
        SetAttribute(Attribute::Reset),
        MoveToColumn(0),
        Clear(ClearType::CurrentLine)
    )?;

    out.flush()
}

#[cfg(test)]
#[path = "../../../../tests/unit/cli_output/progress_view.rs"]
mod tests;
