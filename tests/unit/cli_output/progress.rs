use super::*;

#[test]
fn header_completion_keeps_the_display_live_until_metadata_finishes() {
    let bar = ProgressBar::hidden();
    let mut progress = InspectionProgress {
        animated: true,
        bar: Some(bar.clone()),
    };

    progress.update(ModelEvent::Headers {
        completed: 0,
        total: 2,
        file: None,
    });
    progress.update(ModelEvent::Headers {
        completed: 2,
        total: 2,
        file: Some("weights.safetensors".into()),
    });

    assert_eq!(bar.position(), 2);
    assert_eq!(bar.length(), Some(2));
    assert!(!bar.is_finished());

    progress.update(ModelEvent::ReadingMetadata);

    assert_eq!(bar.length(), None);
    assert!(!bar.is_finished());

    drop(progress);

    assert!(bar.is_finished());
}

#[test]
fn returning_an_error_finishes_the_display() {
    let bar = ProgressBar::hidden();
    let operation = || -> anyhow::Result<()> {
        let mut progress = InspectionProgress {
            animated: true,
            bar: Some(bar.clone()),
        };

        progress.update(ModelEvent::Resolving {
            source: "hf://example/model".into(),
        });

        anyhow::bail!("metadata request failed")
    };

    assert!(operation().is_err());
    assert!(bar.is_finished());
}
