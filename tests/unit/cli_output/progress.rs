use super::*;
use std::sync::mpsc;

#[test]
fn a_cached_lookup_never_draws() {
    let result = drive(|_| Ok(7), |_| panic!("cached lookup drew progress")).unwrap();

    assert_eq!(result, 7);
}

#[test]
fn rendering_ticks_while_inspection_waits_and_preserves_the_error() {
    let caller = std::thread::current().id();
    let (sender, receiver) = mpsc::channel();
    let mut frames = 0;
    let result: Result<()> = drive(
        move |events| {
            events(ModelEvent::Resolving {
                source: "hf://example/model".into(),
            });
            receiver.recv_timeout(Duration::from_secs(2)).unwrap();

            anyhow::bail!("metadata failed")
        },
        |_| {
            assert_eq!(std::thread::current().id(), caller);

            frames += 1;

            if frames == 2 {
                sender.send(()).unwrap();
            }

            Ok(())
        },
    );

    assert!(frames >= 2);
    assert_eq!(result.unwrap_err().to_string(), "metadata failed");
}

#[test]
fn display_failure_drains_events_and_returns_the_operation_result() {
    let (sender, receiver) = mpsc::channel();
    let mut frames = 0;
    let result = drive(
        move |events| {
            events(ModelEvent::Resolving {
                source: "hf://example/model".into(),
            });
            receiver.recv_timeout(Duration::from_secs(2)).unwrap();

            // More events than the queue can hold: the fallback must keep draining.
            for completed in 0..64 {
                events(ModelEvent::Headers {
                    completed,
                    total: 64,
                    file: None,
                });
            }

            Ok(7)
        },
        |_| {
            frames += 1;

            sender.send(()).unwrap();

            Err(io::Error::other("terminal failed"))
        },
    )
    .unwrap();

    assert_eq!(frames, 1);
    assert_eq!(result, 7);
}
