use super::*;
use ratatui::style::Color;

fn headers(completed: usize, total: usize) -> Progress {
    Progress::new(ModelEvent::Headers {
        completed,
        total,
        file: None,
    })
}

fn text(buffer: &Buffer) -> String {
    buffer.content.iter().map(|cell| cell.symbol()).collect()
}

#[test]
fn header_eta_uses_completed_work_and_waits_for_a_sample() {
    let elapsed = Duration::from_secs(10);

    assert_eq!(eta(0, 100, elapsed), None);
    assert_eq!(eta(20, 100, elapsed), Some(40));
    assert_eq!(eta(100, 100, elapsed), Some(0));
    assert_eq!(eta(101, 100, elapsed), Some(0));

    let initial = headers(0, 100).render(80, elapsed);
    let partial = headers(20, 100).render(80, elapsed);

    assert!(text(&initial).contains("ETA --"));
    assert!(text(&partial).contains("headers 20/100 ETA 40s"));
    assert!(partial.content.iter().any(|cell| cell.fg == Color::Cyan));
}

#[test]
fn metadata_stays_visible_after_headers_and_narrow_rows_do_not_wrap() {
    let mut progress = headers(100, 100);

    progress.update(ModelEvent::ReadingMetadata);

    assert!(text(&progress.render(80, Duration::ZERO)).contains("Reading model metadata"));

    for width in [0, 1, 8, 20, 80] {
        for progress in [headers(20, 100), Progress::new(ModelEvent::ReadingMetadata)] {
            let buffer = progress.render(width, Duration::from_secs(1));
            let mut bytes = Vec::new();

            write_row(&mut bytes, &buffer).unwrap();

            assert_eq!(buffer.content.len(), usize::from(width));
            assert!(!bytes.contains(&b'\n'));
            assert!(!bytes.windows(4).any(|part| part == b"\x1b[6n"));
        }
    }
}

#[test]
fn clearing_a_status_row_does_not_clear_the_screen_or_query_the_cursor() {
    let mut bytes = Vec::new();

    clear_row(&mut bytes).unwrap();

    assert_eq!(bytes, b"\x1b[0m\x1b[1G\x1b[2K");
}
