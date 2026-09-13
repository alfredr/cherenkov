use super::*;
use crate::{
    config::Config,
    runner::PrefillResume,
    server::{
        ApiKind,
        output::write_frames,
        registry::Registry,
        response::Response,
        tests::fixtures::{begin_turn, message, new_session},
        tool_call::ToolCallOutputContract,
    },
};
use serde_json::Value;

const CALL: &str =
    "<tool_call><function=weather><parameter=city>Boston</parameter></function></tool_call>";

struct Fixture {
    active: Active<'static>,
    tok: ChatTokenizer,
    receiver: mpsc::Receiver<Frame>,
    store: Arc<Mutex<Store>>,
    session_id: String,
}

fn fixture(raw: &str, tools: bool, reason: &'static str) -> Fixture {
    // One synthetic token decodes to the complete output. Tests feed only a
    // prefix to simulate bytes withheld by the incremental tokenizer.
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab([(raw.to_owned(), 0)].into_iter().collect())
        .build()
        .unwrap();
    let tok = ChatTokenizer {
        inner: tokenizers::Tokenizer::new(model),
        template: Some(crate::prompt::fixture_template()),
        im_end: 1,
        endoftext: 2,
    };
    let config = Config::default();
    let (store, session_id) = new_session(&config, json!({}));
    let ticket = Arc::new(Registry::default())
        .register(Some("worker-test"), 1)
        .unwrap();
    let body = message(&session_id, "Weather?");
    let session = begin_turn(&store, &body, &ticket.id);
    let request = parse_request(
        &body,
        ApiKind::Chat,
        &config.defaults,
        Some(&session.input),
        tok.template.as_ref(),
    )
    .unwrap();
    let options = Options::default();
    let mut decode =
        Decode::new(PrefillResume::default(), &[0], &tok, &options, 128, None).unwrap();
    decode.tokens = vec![0];
    decode.finish_reason = Some(reason);
    let (output, receiver) = Output::for_test(ticket.clone());
    let contract = ToolCallOutputContract::from_tools(&[json!({
        "function": {"name":"weather", "parameters":{
            "type":"object", "properties":{"city":{"type":"string"}}
        }}
    })]);
    let active = Active {
        prepared: PreparedRequest {
            request,
            ids: vec![0],
            boundaries: vec![],
            options,
        },
        phase: Phase::Decode(Box::new(decode)),
        checkpoint: None,
        reservation: 0,
        ticket,
        session: Some(session),
        output,
        text_decoder: Box::new(|_| Ok(None)),
        text: GeneratedText {
            tool_decoder: tools.then(|| ToolCallOutputDecoder::new(contract, true)),
            ..Default::default()
        },
        usage: UsageStats {
            prompt_tokens: 1,
            generated_tokens: 1,
            ..Default::default()
        },
        config_generation: 1,
        response_bytes: 4096,
        last_chunk: None,
    };

    Fixture {
        active,
        tok,
        receiver,
        store,
        session_id,
    }
}

fn read_response(wire: &str, streaming: bool) -> (Value, String) {
    let body = wire.split_once("\r\n\r\n").unwrap().1;

    if !streaming {
        let response: Value = serde_json::from_str(body).unwrap();
        let choice = &response["choices"][0];

        return (
            choice["message"].clone(),
            choice["finish_reason"].as_str().unwrap().into(),
        );
    }

    assert!(body.ends_with("data: [DONE]\n\n"));

    let mut message = json!({"role":"assistant", "content":""});
    let mut content = String::new();
    let mut reason = String::new();

    for data in body.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if data == "[DONE]" {
            break;
        }

        let chunk: Value = serde_json::from_str(data).unwrap();
        let choice = &chunk["choices"][0];
        let delta = &choice["delta"];

        if let Some(text) = delta["content"].as_str() {
            content.push_str(text);
        }

        if let Some(calls) = delta["tool_calls"].as_array() {
            let mut calls = calls.clone();

            for (index, call) in calls.iter_mut().enumerate() {
                assert_eq!(call["index"], index);
                call.as_object_mut().unwrap().remove("index");
            }

            message["tool_calls"] = json!(calls);
        }

        if let Some(finish) = choice["finish_reason"].as_str() {
            assert_eq!(delta, &json!({}));

            reason = finish.to_owned();
        }
    }

    message["content"] = if content.is_empty() && message.get("tool_calls").is_some() {
        Value::Null
    } else {
        json!(content)
    };

    (message, reason)
}

fn complete(
    raw: &str,
    split: usize,
    tools: bool,
    reason: &'static str,
    streaming: bool,
) -> (Value, String) {
    let mut fixture = fixture(raw, tools, reason);
    let active = &mut fixture.active;

    active
        .text
        .feed(&raw[..split], &active.output, active.response_bytes)
        .unwrap();

    finish_fixture(fixture, streaming)
}

fn finish_fixture(fixture: Fixture, streaming: bool) -> (Value, String) {
    let Fixture {
        active,
        tok,
        receiver,
        store,
        session_id,
    } = fixture;
    let ticket = active.ticket.clone();
    let generated_tokens = active.usage.generated_tokens;

    active.finish(&tok).unwrap();

    // Finalization prepares history; only the response writer publishes it.
    assert_eq!(
        store.lock().unwrap().show(&session_id).unwrap()["messages"],
        json!([])
    );

    let mut bytes = Vec::new();
    let mut response =
        Response::for_writer(&mut bytes, ApiKind::Chat, &ticket.id, 0, streaming, false);

    assert!(write_frames(&mut response, receiver, &ticket).unwrap());

    let (mut message, reason) = read_response(&String::from_utf8(bytes).unwrap(), streaming);

    // Session arguments are decoded objects; the wire carries JSON strings.
    if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
        for call in calls {
            let arguments = call["function"]["arguments"].as_str().unwrap();
            call["function"]["arguments"] = serde_json::from_str(arguments).unwrap();
        }
    }

    let session = store.lock().unwrap().show(&session_id).unwrap();

    assert_eq!(session["messages"][1], message);
    assert_eq!(
        session["stats"]["usage"]["generated_tokens"],
        generated_tokens
    );

    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        let prompt = tok
            .template
            .as_ref()
            .unwrap()
            .chat(session["messages"].as_array().unwrap(), None)
            .unwrap();

        assert_eq!(
            prompt.text.matches("<function=weather>").count(),
            calls.len()
        );
    }

    (message, reason)
}

fn complete_bytelevel(raw: &str, truncate: bool, streaming: bool) -> (Value, String) {
    use tokenizers::{
        models::bpe::{BPE, Vocab},
        pre_tokenizers::byte_level::ByteLevel,
    };

    // No merges: each token represents one byte, so accented letters and
    // emoji require several tokens before decode_stream can publish them.
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();

    alphabet.sort_unstable();

    let model = BPE::builder()
        .vocab_and_merges(
            alphabet
                .iter()
                .enumerate()
                .map(|(id, c)| (c.to_string(), id as u32))
                .collect::<Vocab>(),
            vec![],
        )
        .build()
        .unwrap();
    let mut fixture = fixture(raw, true, if truncate { "length" } else { "stop" });
    fixture.tok.inner = tokenizers::Tokenizer::new(model);

    fixture
        .tok
        .inner
        .with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    fixture.tok.inner.with_decoder(Some(ByteLevel::default()));

    let mut ids = fixture.tok.encode(raw).unwrap();

    assert_eq!(ids.len(), raw.len());
    assert_eq!(fixture.tok.decode(&ids).unwrap(), raw);

    if truncate {
        // Stop partway through the final emoji's UTF-8 encoding.
        ids.pop();
    }

    let active = &mut fixture.active;
    let mut decoder = fixture.tok.inner.decode_stream(false);
    let mut withheld = 0;

    for &id in &ids {
        match decoder.step(id).unwrap() {
            Some(delta) => active
                .text
                .feed(&delta, &active.output, active.response_bytes)
                .unwrap(),
            None => withheld += 1,
        }
    }

    assert!(
        withheld > 0,
        "the real decoder must hold incomplete UTF-8 bytes"
    );

    let full = fixture.tok.decode(&ids).unwrap();
    let expected_tail = if truncate { "�" } else { "" };

    assert_eq!(full.strip_prefix(&active.text.raw), Some(expected_tail));

    let Phase::Decode(decode) = &mut active.phase else {
        unreachable!()
    };
    active.usage.generated_tokens = ids.len() as u64;
    decode.tokens = ids;

    finish_fixture(fixture, streaming)
}

#[test]
fn bytelevel_tokens_preserve_unicode_content_and_structured_calls() {
    let raw = format!("Café. {CALL}");

    for streaming in [false, true] {
        let (message, reason) = complete_bytelevel(&raw, false, streaming);

        assert_eq!(message["content"], "Café.");
        assert_eq!(message["tool_calls"].as_array().unwrap().len(), 1);
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            json!({"city":"Boston"})
        );
        assert_eq!(reason, "tool_calls");
    }
}

#[test]
fn bytelevel_truncation_reconciles_withheld_bytes_before_tool_finalization() {
    let malformed = "Café. <tool_call><function=weather><parameter=city>☕";
    let recovered = format!("Café. {CALL}<tool_call><function=weather><parameter=city>☕");
    let cases = [
        ("Café ☕", "Café �".to_owned(), false),
        (malformed, malformed.replace('☕', "�"), false),
        (recovered.as_str(), "Café.".to_owned(), true),
    ];

    for (raw, content, calls) in cases {
        for streaming in [false, true] {
            let (message, reason) = complete_bytelevel(raw, true, streaming);

            assert_eq!(message["content"], content);
            assert_eq!(message.get("tool_calls").is_some(), calls);
            assert_eq!(reason, "length");
        }
    }
}

#[test]
fn parsed_content_matches_stream_json_and_session_at_every_split() {
    for prefix in ["", "  Checking Montréal.\n"] {
        let raw = format!("{prefix}{CALL}");

        for split in (0..=raw.len()).filter(|&n| raw.is_char_boundary(n)) {
            for streaming in [false, true] {
                let (message, reason) = complete(&raw, split, true, "stop", streaming);
                let expected = if prefix.is_empty() {
                    Value::Null
                } else {
                    json!(prefix.trim_end())
                };

                assert_eq!(message["content"], expected);
                assert_eq!(reason, "tool_calls");
                assert_eq!(message["tool_calls"].as_array().unwrap().len(), 1);
                assert_eq!(
                    message["tool_calls"][0]["function"]["arguments"],
                    json!({"city":"Boston"})
                );
            }
        }
    }
}

#[test]
fn fallback_and_plain_text_reach_all_clients_verbatim() {
    for raw in [
        "",
        "Plain Montréal. \n",
        "Hi <tool_ca",
        "Hi <tool_call><function=unknown></function></tool_call>",
        "Hi <tool_call><function=weather><parameter=city>unfinished",
        CALL,
    ] {
        for tools in [false, true] {
            if tools && raw == CALL {
                continue;
            }

            for split in [0, raw.len() / 2, raw.len()] {
                for streaming in [false, true] {
                    let (message, reason) = complete(raw, split, tools, "stop", streaming);

                    assert_eq!(message["content"], raw);
                    assert!(message.get("tool_calls").is_none());
                    assert_eq!(reason, "stop");
                }
            }
        }
    }
}

#[test]
fn recovered_calls_preserve_length_and_discard_only_the_tool_suffix() {
    let truncated = "Checking. <tool_call><function=weather><parameter=city>Boston</parameter>";
    let suffix = format!("Checking. {CALL}trailing text");
    let next_call =
        format!("Checking. {CALL}<tool_call><function=weather><parameter=city>unfinished");

    for raw in [truncated, &suffix, &next_call] {
        for stopped in ["length", "stop"] {
            for streaming in [false, true] {
                let (message, reason) = complete(raw, raw.len(), true, stopped, streaming);

                assert_eq!(message["content"], "Checking.");
                assert_eq!(message["tool_calls"].as_array().unwrap().len(), 1);
                assert_eq!(
                    reason,
                    if stopped == "stop" {
                        "tool_calls"
                    } else {
                        "length"
                    }
                );
            }
        }
    }
}

#[test]
fn raw_tool_bytes_obey_the_response_limit_before_finalization() {
    let Fixture { mut active, .. } = fixture(CALL, true, "stop");
    active.response_bytes = CALL.len() - 1;

    let error = active
        .text
        .feed(CALL, &active.output, active.response_bytes)
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("response exceeds response_bytes")
    );
}

#[test]
fn final_tokenizer_bytes_cannot_exceed_the_response_limit_or_commit_history() {
    let Fixture {
        mut active,
        tok,
        receiver,
        store,
        session_id,
    } = fixture(CALL, true, "stop");
    active.response_bytes = CALL.len() - 1;

    let error = active.finish(&tok).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("response exceeds response_bytes")
    );
    assert!(matches!(receiver.recv().unwrap(), Frame::Error(_)));
    assert_eq!(
        store.lock().unwrap().show(&session_id).unwrap()["messages"],
        json!([])
    );
}

#[test]
fn cancellation_after_preparation_does_not_commit_parsed_content() {
    let Fixture {
        active,
        tok,
        receiver,
        store,
        session_id,
    } = fixture(CALL, true, "length");
    let ticket = active.ticket.clone();

    active.finish(&tok).unwrap();
    ticket.cancel();

    let mut bytes = Vec::new();
    let mut response = Response::for_writer(&mut bytes, ApiKind::Chat, &ticket.id, 0, true, false);

    assert!(!write_frames(&mut response, receiver, &ticket).unwrap());
    assert_eq!(
        read_response(&String::from_utf8(bytes).unwrap(), true).1,
        "cancelled"
    );
    assert_eq!(
        store.lock().unwrap().show(&session_id).unwrap()["messages"],
        json!([])
    );
}
