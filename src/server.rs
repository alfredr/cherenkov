//! Server startup and shared job ownership. One worker owns the loaded GPU.

use crate::{
    config::Source,
    control::{self, State, state::Versioned},
    prefix_cache::PrefixCache,
    qwen4_exp,
    tok::ChatTokenizer,
    units::BYTES_PER_GB,
};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::net::{TcpListener, TcpStream};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

mod failure;
mod http;
mod output;
mod pacer;
mod registry;
mod request;
mod response;
mod routes;
mod sessions;
mod stats;
mod tool_call;
mod worker;

pub use stats::UsageStats;

use http::error;
use routes::connection;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ApiKind {
    Chat,
    Completion,
}

const MODEL: &str = "cherenkov";

struct Job {
    stream: TcpStream,
    body: Value,
    kind: ApiKind,
    settings: Versioned,
    ticket: Arc<registry::Ticket>,
    session: Option<sessions::Turn>,
}

pub fn serve(source: Source) -> Result<()> {
    let config = source.resolve()?;
    let mut options = config.options();
    options.repack = source.overrides.repack;

    ensure!(
        !options.repack || options.build_missing_store,
        "--repack conflicts with build_missing_store=false"
    );
    options.validate()?;

    // Retain the lease until the worker and all GPU resources have been dropped.
    let model =
        crate::model::index::resolve_runtime(config.paths()?, &config.model_dir()?, &mut options)?;
    let model_dir = model.path.clone();

    let cache_bytes = config.cache_bytes();
    let cache = PrefixCache::new(
        cache_bytes,
        config.limits.cache_max_entries,
        config.limits.cache_idle_seconds,
    );
    let state = Arc::new(State::new(source, config.clone()));
    let requests = Arc::new(registry::Registry::default());
    let sessions = Arc::new(std::sync::Mutex::new(sessions::Store::new(&config)));
    let _control = control::Listener::start(&config.server.socket, state.clone())?;

    eprintln!("control socket: {}", config.server.socket.display());

    let listener =
        TcpListener::bind(("127.0.0.1", config.server.port)).context("binding server")?;
    let address = listener.local_addr()?;
    let tok = ChatTokenizer::load(&model_dir)?;
    let packed = qwen4_exp::packed::Packed::open(&model_dir)?;
    let gpu = qwen4_exp::gpu::Gpu::load_bounded(
        &packed,
        options.max_ctx,
        &options,
        config.reserved_bytes(),
        Some(config.memory_bytes()),
        config.limits.prefill_quantum,
    )?;

    ensure!(
        gpu.allocated_gb() + config.reserved_bytes() as f64 / BYTES_PER_GB as f64
            <= config.limits.memory_gb,
        "Metal plus cache/session reservations exceeds configured memory budget"
    );
    state.observe(&gpu, &cache, &mut Default::default());
    state.update(|s| {
        s.ready = true;
        s.http_address = Some(address.to_string());
    });

    if options.cut_weak > 0.0 {
        eprintln!("WARNING: --cut-weak makes output depend on disk timing and non-reproducible.");
    }

    eprintln!(
        "cherenkov serving http://{address}/v1, model {MODEL}, {:.2} GB Metal",
        gpu.allocated_gb()
    );

    let (tx, rx) = mpsc::sync_channel::<Job>(config.limits.queued_requests);
    let connections = state.clone();
    let network_sessions = sessions.clone();

    std::thread::spawn(move || {
        let active = Arc::new(AtomicUsize::new(0));

        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };

            if active.fetch_add(1, Ordering::AcqRel) >= config.limits.http_readers {
                active.fetch_sub(1, Ordering::AcqRel);
                stream.set_write_timeout(Some(Duration::from_secs(1))).ok();
                connections.update(|s| s.rejected_requests += 1);

                let _ = error(&mut stream, 503, "too many connections");

                continue;
            }

            let tx = tx.clone();
            let active = active.clone();
            let state = connections.clone();
            let requests = requests.clone();
            let sessions = network_sessions.clone();

            std::thread::spawn(move || {
                if let Err(e) = connection(stream, &tx, &state, &requests, &sessions) {
                    eprintln!("HTTP connection: {e:#}");
                }

                active.fetch_sub(1, Ordering::AcqRel);
            });
        }
    });
    worker::Worker::new(gpu, &tok, options, cache, state, sessions).run(rx);

    Ok(())
}

#[cfg(test)]
#[path = "../tests/unit/server.rs"]
mod tests;
