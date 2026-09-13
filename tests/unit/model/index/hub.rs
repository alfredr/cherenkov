use super::*;
use std::{
    net::TcpListener,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

struct Fixture {
    endpoint: String,
    reads: Arc<Mutex<Vec<(String, u64, u64)>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Fixture {
    fn new(ignore_range: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();

        listener.set_nonblocking(true).unwrap();

        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let reads = Arc::new(Mutex::new(Vec::new()));
        let recorded = reads.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || {
            while !stopping.load(Ordering::Relaxed) {
                let Ok((mut stream, _)) = listener.accept() else {
                    thread::sleep(Duration::from_millis(2));

                    continue;
                };

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
                let (status, extra, body) = response(path, &request, &recorded, ignore_range);

                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n",
                    body.len()
                )
                .unwrap();
                stream.write_all(&body).unwrap();
            }
        });

        Self {
            endpoint,
            reads,
            stop,
            worker: Some(worker),
        }
    }

    fn hub(&self) -> Hub {
        Hub {
            client: Client::new(),
            endpoint: self.endpoint.clone(),
            token: Some("test-token".into()),
        }
    }
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
        return json_response(
            json!({"weight_map":{"one":"one.safetensors","two":"two.safetensors"}}),
        );
    }

    let name = if path.contains("one.safetensors") {
        "one"
    } else {
        "two"
    };
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
    let (source, description) = fixture.hub().inspect("example/tiny", "moving-tag").unwrap();
    let Source::HuggingFace { revision, .. } = source else {
        panic!("HF source expected")
    };

    assert_eq!(revision, "a".repeat(40));
    assert_eq!(description.tensors.len(), 2);

    let reads = fixture.reads.lock().unwrap();

    assert!(reads.iter().any(|(name, _, _)| name == "one"));
    assert!(reads.iter().any(|(name, _, _)| name == "two"));
    assert_eq!(reads.len(), 4);
}

#[test]
fn ignored_byte_ranges_do_not_trigger_full_weight_downloads() {
    let fixture = Fixture::new(true);
    let error = fixture.hub().inspect("example/tiny", "main").err().unwrap();

    assert!(error.to_string().contains("refusing a full shard read"));
}
