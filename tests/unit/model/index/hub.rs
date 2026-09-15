use super::*;
use std::{
    net::TcpListener,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

#[derive(Default)]
struct Activity {
    requests: usize,
    active: usize,
    peak: usize,
    prefixes: usize,
    header_reported: bool,
    timed_out: bool,
}

#[derive(Default)]
struct Monitor {
    activity: Mutex<Activity>,
    changed: std::sync::Condvar,
}

impl Monitor {
    /// Hold the first wave until enough requests overlap; a serial regression times out.
    fn started(&self, request: &str, wave: usize) -> bool {
        let mut activity = self.activity.lock().unwrap();
        let range = request.to_lowercase().contains("range: bytes=");

        activity.requests += 1;

        if !range {
            return false;
        }

        activity.active += 1;
        activity.peak = activity.peak.max(activity.active);

        if wave == 0 {
            return true;
        }

        // Make shard zero finish later, so file-order assertions exercise reordering.
        if request.contains("/tensor000.safetensors?")
            && request.to_lowercase().contains("range: bytes=8-")
        {
            let (mut activity, timeout) = self
                .changed
                .wait_timeout_while(activity, Duration::from_secs(2), |a| !a.header_reported)
                .unwrap();

            activity.timed_out |= timeout.timed_out();

            return true;
        }

        if !request.to_lowercase().contains("range: bytes=0-7") {
            return true;
        }

        activity.prefixes += 1;

        self.changed.notify_all();

        let (mut activity, timeout) = self
            .changed
            .wait_timeout_while(activity, Duration::from_secs(2), |a| a.prefixes < wave)
            .unwrap();

        activity.timed_out |= timeout.timed_out();

        true
    }
}

struct Fixture {
    endpoint: String,
    reads: Arc<Mutex<Vec<(String, u64, u64)>>>,
    monitor: Arc<Monitor>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Fixture {
    fn new(ignore_range: bool) -> Self {
        Self::with_shards(ignore_range, 2, 0)
    }

    fn with_shards(ignore_range: bool, shards: usize, wave: usize) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();

        listener.set_nonblocking(true).unwrap();

        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let reads = Arc::new(Mutex::new(Vec::new()));
        let recorded = reads.clone();
        let monitor = Arc::new(Monitor::default());
        let observed = monitor.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || {
            thread::scope(|scope| {
                while !stopping.load(Ordering::Relaxed) {
                    let Ok((stream, _)) = listener.accept() else {
                        thread::sleep(Duration::from_millis(2));

                        continue;
                    };
                    let observed = &observed;
                    let recorded = &recorded;

                    scope.spawn(move || {
                        serve(stream, recorded, observed, ignore_range, shards, wave)
                    });
                }
            })
        });

        Self {
            endpoint,
            reads,
            monitor,
            stop,
            worker: Some(worker),
        }
    }

    fn hub(&self) -> Hub {
        Hub {
            client: Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            endpoint: self.endpoint.clone(),
            token: Some("test-token".into()),
        }
    }
}

fn serve(
    mut stream: std::net::TcpStream,
    reads: &Mutex<Vec<(String, u64, u64)>>,
    monitor: &Monitor,
    ignore_range: bool,
    shards: usize,
    wave: usize,
) {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();

    let mut request = Vec::new();
    let mut byte = [0];

    while !request.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).unwrap() == 0 {
            break;
        }

        request.push(byte[0]);
    }

    let request = String::from_utf8(request).unwrap();
    let path = request.split_whitespace().nth(1).unwrap();
    let active = monitor.started(&request, wave);
    let (status, extra, body) = response(path, &request, reads, ignore_range, shards);

    if active {
        monitor.activity.lock().unwrap().active -= 1;
    }

    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n",
        body.len()
    )
    .unwrap();
    stream.write_all(&body).unwrap();
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);

        let result = self.worker.take().unwrap().join();

        if !thread::panicking() {
            result.unwrap();
        }
    }
}

fn response(
    path: &str,
    request: &str,
    reads: &Mutex<Vec<(String, u64, u64)>>,
    ignore_range: bool,
    shards: usize,
) -> (&'static str, String, Vec<u8>) {
    assert!(
        request
            .to_lowercase()
            .contains("authorization: bearer test-token")
    );

    let revision = "a".repeat(40);

    if path.starts_with("/api/") {
        return json_response(
            json!({"sha":revision,"siblings":[{"rfilename":"model.safetensors.index.json"}]}),
        );
    }

    assert!(path.starts_with(&format!("/example/tiny/resolve/{revision}/")));

    if path.ends_with("config.json") {
        return json_response(json!({"model_type":"example"}));
    }

    if path.ends_with("model.safetensors.index.json") {
        let weights: serde_json::Map<String, Value> = (0..shards)
            .map(|id| {
                (
                    format!("tensor{id:03}"),
                    json!(format!("tensor{id:03}.safetensors")),
                )
            })
            .collect();

        return json_response(json!({"weight_map": weights}));
    }

    let name = path.rsplit('/').next().unwrap().split('.').next().unwrap();
    let header =
        serde_json::to_vec(&json!({name:{"dtype":"F32","shape":[1024],"data_offsets":[0,4096]}}))
            .unwrap();
    let payload_start = 8 + header.len() as u64;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();

    bytes.extend(header);
    bytes.resize(bytes.len() + 4096, 0);

    let lower = request.to_lowercase();
    let range = lower
        .lines()
        .find_map(|l| l.strip_prefix("range: bytes="))
        .unwrap();
    let (start, end) = range.split_once('-').unwrap();
    let (start, end): (u64, u64) = (start.trim().parse().unwrap(), end.trim().parse().unwrap());

    assert!(end < payload_start, "registration read weight payload");
    reads.lock().unwrap().push((name.into(), start, end));

    if ignore_range {
        return ("200 OK", String::new(), bytes);
    }

    (
        "206 Partial Content",
        format!("Content-Range: bytes {start}-{end}/{}\r\n", bytes.len()),
        bytes[start as usize..=end as usize].to_vec(),
    )
}

fn json_response(value: Value) -> (&'static str, String, Vec<u8>) {
    ("200 OK", String::new(), serde_json::to_vec(&value).unwrap())
}

#[test]
fn registration_pins_revision_and_crawls_each_shard_without_weights() {
    let fixture = Fixture::new(false);
    let (source, description) = fixture
        .hub()
        .inspect("example/tiny", "moving-tag", &mut |_| {})
        .unwrap();
    let Source::HuggingFace { revision, .. } = source else {
        panic!("HF source expected")
    };

    assert_eq!(revision, "a".repeat(40));
    assert_eq!(description.tensors.len(), 2);

    let reads = fixture.reads.lock().unwrap();

    assert!(reads.iter().any(|(name, _, _)| name == "tensor000"));
    assert!(reads.iter().any(|(name, _, _)| name == "tensor001"));
    assert_eq!(reads.len(), 4);
}

#[test]
fn ignored_byte_ranges_do_not_trigger_full_weight_downloads() {
    let fixture = Fixture::with_shards(true, 19, 0);
    let mut events = Vec::new();
    let error = fixture
        .hub()
        .inspect("example/tiny", "main", &mut |event| events.push(event))
        .err()
        .unwrap();

    assert!(format!("{error:#}").contains("refusing a full shard read"));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, ModelEvent::ReadingMetadata))
    );

    let reads = fixture.reads.lock().unwrap();

    assert!(!reads.is_empty());
    assert!(
        reads.len() <= 8,
        "workers kept claiming shards after failure"
    );
    assert!(reads.iter().all(|(_, start, end)| (*start, *end) == (0, 7)));
}

#[test]
fn parallel_headers_are_bounded_and_events_run_on_the_caller_thread() {
    let shards = 19;
    let fixture = Fixture::with_shards(false, shards, 8);
    let caller = thread::current().id();
    let mut events = Vec::new();
    let (_, description) = fixture
        .hub()
        .inspect("example/tiny", "main", &mut |event| {
            assert_eq!(thread::current().id(), caller);

            match &event {
                ModelEvent::Resolving { .. } => {
                    assert_eq!(fixture.monitor.activity.lock().unwrap().requests, 0)
                }
                ModelEvent::Headers { completed: 0, .. } => {
                    assert!(fixture.reads.lock().unwrap().is_empty())
                }
                ModelEvent::Headers { .. } => {
                    fixture.monitor.activity.lock().unwrap().header_reported = true;

                    fixture.monitor.changed.notify_all();
                }
                _ => {}
            }

            events.push(event);
        })
        .unwrap();
    let activity = fixture.monitor.activity.lock().unwrap();

    assert!(!activity.timed_out, "header requests did not overlap");
    assert_eq!(
        activity.peak, 8,
        "requests exceeded the bound or ran serially"
    );
    assert_eq!(fixture.reads.lock().unwrap().len(), shards * 2);

    let reads = fixture.reads.lock().unwrap();
    let first_header = reads.iter().position(|(_, start, _)| *start == 8).unwrap();

    assert!(first_header < shards, "prefix reads formed a separate pass");
    assert_eq!(description.tensors.len(), shards);

    for (id, tensor) in description.tensors.iter().enumerate() {
        assert_eq!(tensor.name, format!("tensor{id:03}"));
        assert_eq!(tensor.encoding.data()[0].object, ObjectId(id));
    }

    assert_header_events(&events, shards);
}

/// Header completions are monotonic even when workers finish out of file order.
fn assert_header_events(events: &[ModelEvent], shards: usize) {
    let counts: Vec<_> = events
        .iter()
        .filter_map(|event| {
            if let ModelEvent::Headers {
                completed,
                total,
                file,
            } = event
            {
                assert_eq!(*total, shards);
                assert_eq!(file.is_none(), *completed == 0);

                return Some(*completed);
            }

            None
        })
        .collect();

    assert_eq!(counts, (0..=shards).collect::<Vec<_>>());
    assert!(
        matches!(&events[2], ModelEvent::Headers { file: Some(file), .. } if file != "tensor000.safetensors")
    );
    assert_eq!(events.last(), Some(&ModelEvent::ReadingMetadata));
}
