//! Live checks are explicit: they need a packed model or network access.
#[path = "smoke/concurrency.rs"]
mod concurrency;
mod support;

use anyhow::Result;
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    net::TcpStream,
    process::Command,
    time::{Duration, Instant},
};
use support::*;

#[test]
#[ignore = "requires the real model and Metal"]
fn server_smoke() -> Result<()> {
    let server = Server::start(900)?;
    let address = &server.address;

    check_discovery(address)?;

    let mut messages = json!([
        {"role": "system", "content": "You are a concise assistant. ".repeat(16)},
        {"role": "user", "content": "What is the capital of France?"},
    ]);
    let first = std::thread::scope(|scope| -> Result<Value> {
        let pending = scope.spawn(|| chat(address, &messages));

        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(
            request(address, "/health", None)?.success()?["status"],
            "ok"
        );

        pending.join().unwrap()
    })?;

    assert_eq!(first["object"], "chat.completion");
    assert!(matches!(
        first["choices"][0]["finish_reason"].as_str(),
        Some("stop" | "length")
    ));
    assert_eq!(first["usage"]["prompt_tokens_details"]["cached_tokens"], 0);

    let second = chat(address, &messages)?;

    assert_eq!(
        first["choices"][0]["message"]["content"],
        second["choices"][0]["message"]["content"]
    );
    assert_eq!(
        second["usage"]["prompt_tokens_details"]["cached_tokens"],
        second["usage"]["prompt_tokens"]
    );

    messages[1]["content"] = json!("What is the capital of Japan?");
    let third = chat(address, &messages)?;
    let cached = third["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();

    assert!(cached > 0 && cached < third["usage"]["prompt_tokens"].as_u64().unwrap());

    let mut conversation = messages.clone();

    conversation.as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":third["choices"][0]["message"]["content"]}),
        json!({"role":"user","content":"And France?"}),
    ]);

    let turn = chat(address, &conversation)?;

    assert!(
        turn["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > cached
    );

    let raw = complete(
        address,
        "The capital of Italy is",
        json!({"max_tokens":4,"temperature":0}),
    )?
    .success()?;

    assert_eq!(raw["object"], "text_completion");
    assert!(raw["choices"][0]["text"].is_string());

    let raw2 = complete(
        address,
        "The capital of Italy is Rome",
        json!({"max_tokens":4}),
    )?
    .success()?;

    assert!(
        raw2["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    chat(address, &messages)?;
    check_stream(address, &messages)?;
    check_invalid_requests(address, &messages)?;
    save_result(
        "server",
        json!({"exact_repeat":second["usage"],"shared_system":third["usage"],"next_chat_turn":turn["usage"],"extended_raw":raw2["usage"],"stream":"verified","invalid_requests":"400 responses verified"}),
    )?;

    Ok(())
}

fn check_discovery(address: &str) -> Result<()> {
    assert_eq!(
        request(address, "/v1/models", None)?.success()?["data"][0]["id"],
        "cherenkov"
    );

    let parked = TcpStream::connect(address)?;
    let start = Instant::now();

    assert_eq!(
        request(address, "/health", None)?.success()?["status"],
        "ok"
    );
    assert!(start.elapsed() < Duration::from_secs(2));
    drop(parked);

    Ok(())
}

fn check_stream(address: &str, messages: &Value) -> Result<()> {
    let response = request(
        address,
        "/v1/chat/completions",
        Some(
            &json!({"model":"cherenkov","messages":messages,"max_tokens":4,"stream":true,"stream_options":{"include_usage":true}}),
        ),
    )?;

    assert_eq!(response.status, 200);
    assert!(
        response
            .headers
            .to_ascii_lowercase()
            .contains("content-type: text/event-stream")
    );

    let events: Vec<_> = response
        .body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();

    assert_eq!(events.last(), Some(&"[DONE]"));

    let chunks = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str::<Value>(e))
        .collect::<std::result::Result<Vec<_>, _>>()?;

    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(matches!(
        chunks[chunks.len() - 2]["choices"][0]["finish_reason"].as_str(),
        Some("stop" | "length")
    ));
    assert_eq!(chunks.last().unwrap()["choices"], json!([]));
    assert!(
        chunks.last().unwrap()["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );

    Ok(())
}

fn check_invalid_requests(address: &str, messages: &Value) -> Result<()> {
    for body in [
        json!({"messages":[]}),
        json!({"messages":messages,"temperature":-0.8}),
        json!({"messages":messages,"max_tokens":100000}),
    ] {
        let response = request(address, "/v1/chat/completions", Some(&body))?;

        assert_eq!(response.status, 400);
        assert!(response.json()?.get("error").is_some());
    }

    Ok(())
}

#[test]
#[ignore = "requires the real model and Metal"]
fn control_smoke() -> Result<()> {
    let server = Server::start(2)?;
    let prompt = "Explain briefly why cache locality matters. ".to_owned()
        + &"A cache stores recently used data for later access. ".repeat(12);
    let complete_request = || complete(&server.address, &prompt, json!({}))?.success();
    let (a, b) = std::thread::scope(|scope| -> Result<_> {
        let first = scope.spawn(complete_request);

        wait_for(
            || Ok(server.stats()?["active_requests"] == 1),
            Duration::from_secs(180),
        )?;

        let second = scope.spawn(complete_request);

        wait_for(
            || Ok(server.stats()?["queued_requests"] == 1),
            Duration::from_secs(180),
        )?;
        server.rewrite(2)?;
        assert_eq!(server.control(&["config", "reload"])?["generation"], 2);

        Ok((first.join().unwrap()?, second.join().unwrap()?))
    })?;
    let c = complete_request()?;
    let counts = [&a, &b, &c].map(|r| r["usage"]["completion_tokens"].as_u64().unwrap());

    assert_eq!(counts, [4, 4, 2]);

    let hits = b["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();

    assert!(hits > 0);

    let before = server.control(&["config", "show"])?["effective"].clone();

    server.rewrite(3)?;

    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(server.directory.path().join("server.toml"))?;

    writeln!(file, "\n[experts]\nmiss_bits = 2")?;

    let error = server.control(&["config", "reload"]).unwrap_err();

    assert!(
        format!("{error:#}").contains("restart required"),
        "{error:#}"
    );
    assert_eq!(server.control(&["config", "show"])?["effective"], before);
    assert_eq!(
        complete(
            &server.address,
            &prompt,
            json!({"max_tokens":MAX_OUTPUT_TOKENS + 1})
        )?
        .status,
        400
    );
    wait_for(
        || Ok(server.stats()?["cache"]["entries"] == 0),
        Duration::from_secs(10),
    )?;

    let stats = server.stats()?;

    assert_eq!(stats["completed_requests"], 3);
    assert_eq!(stats["failed_requests"], 1);
    assert_eq!(stats["generated_tokens"], 10);
    save_result(
        "control",
        json!({"captured_token_limits":counts,"cached_prompt_tokens":hits,"restart_change_rejected":true,"expired_cache":stats["cache"],"memory":stats["memory"]}),
    )?;

    Ok(())
}

#[test]
#[ignore = "downloads metadata from Hugging Face"]
fn download_smoke() -> Result<()> {
    use std::os::unix::fs::MetadataExt;

    let root = tempfile::tempdir()?;
    let binary = binary();
    let output = Command::new(&binary)
        .args(["paths", "--root"])
        .arg(root.path())
        .output()?;
    let paths: Value = serde_json::from_str(&xtask::util::checked(output)?)?;
    let mut stamps = Vec::new();

    for _ in 0..2 {
        let mut command = Command::new(&binary);

        command
            .args(["download", "--root"])
            .arg(root.path())
            .arg("--metadata-only");

        let model = std::path::PathBuf::from(timed_command(command, Duration::from_secs(180))?);

        assert_eq!(model.to_str(), paths["downloaded_model"].as_str());

        let config = xtask::util::json(&model.join("config.json"))?;

        assert_eq!(config["model_type"], "qwen4_exp");
        assert_eq!(config["text_config"]["model_type"], "qwen4_exp_text");
        assert!(model.join("tokenizer.json").is_symlink());

        let mut stamp = std::collections::BTreeMap::new();

        for entry in fs::read_dir(model)? {
            let entry = entry?;

            assert_ne!(
                entry.path().extension().and_then(|e| e.to_str()),
                Some("safetensors")
            );

            let metadata = entry.path().metadata()?;

            if metadata.is_file() {
                stamp.insert(
                    entry.file_name(),
                    (
                        metadata.ino(),
                        metadata.mtime(),
                        metadata.mtime_nsec(),
                        metadata.len(),
                    ),
                );
            }
        }

        stamps.push(stamp);
    }

    assert_eq!(stamps[0], stamps[1]);
    save_result(
        "download",
        json!({"metadata_files":stamps[1].len(),"cached_blobs_reused":true,"weight_shards_downloaded":0}),
    )?;

    Ok(())
}

fn save_result(name: &str, result: Value) -> Result<()> {
    let directory = xtask::util::root().join("results");

    fs::create_dir_all(&directory)?;
    xtask::util::write_json(&directory.join(format!("{name}-smoke.json")), &result)?;
    println!("{}", serde_json::to_string_pretty(&result)?);

    Ok(())
}
