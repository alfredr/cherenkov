use super::*;

#[test]
fn response_wrapper_preserves_endpoint_and_stream_formats() {
    // Full response expectations keep protocol field omissions visible: chat has
    // no logprobs/text, completion has no message/delta, and only chat sends a role.
    let cases = [
        (
            ApiKind::Chat,
            "chatcmpl-123-7",
            "chat.completion",
            "chat.completion.chunk",
            json!({"index":0,"message":{"role":"assistant","content":"Hello \u{e9}"},"finish_reason":"stop"}),
            vec![
                json!({"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}),
                json!({"index":0,"delta":{"content":"Hello "},"finish_reason":null}),
                json!({"index":0,"delta":{"content":"\u{e9}"},"finish_reason":null}),
                json!({"index":0,"delta":{},"finish_reason":"stop"}),
            ],
        ),
        (
            ApiKind::Completion,
            "cmpl-123-7",
            "text_completion",
            "text_completion",
            json!({"index":0,"text":"Hello \u{e9}","logprobs":null,"finish_reason":"stop"}),
            vec![
                json!({"index":0,"text":"Hello ","logprobs":null,"finish_reason":null}),
                json!({"index":0,"text":"\u{e9}","logprobs":null,"finish_reason":null}),
                json!({"index":0,"text":"","logprobs":null,"finish_reason":"stop"}),
            ],
        ),
    ];
    let usage = json!({"prompt_tokens":5,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens":3,"total_tokens":8});

    for (kind, id, object, chunk_object, final_choice, chunks) in cases {
        for streaming in [false, true] {
            for include_usage in [false, true] {
                let mut output = Vec::new();
                let mut response =
                    Response::for_writer(&mut output, kind, id, 123, streaming, include_usage);

                response.start().unwrap();
                response.text("Hello ").unwrap();
                response.text("\u{e9}").unwrap();
                response
                    .finish("Hello \u{e9}", None, "stop", usage.clone())
                    .unwrap();

                let wire = String::from_utf8(output).unwrap();
                let (headers, body) = wire.split_once("\r\n\r\n").unwrap();

                assert!(headers.starts_with("HTTP/1.1 200 OK\r\n"));

                if streaming {
                    assert!(headers.contains("Content-Type: text/event-stream"));

                    let expected =
                        expected_stream(id, chunk_object, &chunks, include_usage.then_some(&usage));

                    assert_eq!(body, expected);
                } else {
                    assert!(headers.contains("Content-Type: application/json"));
                    assert!(headers.contains(&format!("Content-Length: {}\r\n", body.len())));
                    assert_eq!(
                        serde_json::from_str::<Value>(body).unwrap(),
                        json!({"id":id,"object":object,"created":123,"model":"cherenkov","choices":[final_choice],"usage":usage})
                    );
                }
            }
        }
    }
}

fn expected_stream(
    id: &str,
    chunk_object: &str,
    chunks: &[Value],
    usage: Option<&Value>,
) -> String {
    let mut expected = String::new();

    for choice in chunks {
        let chunk = json!({"id":id,"object":chunk_object,"created":123,"model":"cherenkov","choices":[choice]});

        expected.push_str(&format!("data: {chunk}\n\n"));
    }

    if let Some(usage) = usage {
        let chunk = json!({"id":id,"object":chunk_object,"created":123,"model":"cherenkov","choices":[],"usage":usage});

        expected.push_str(&format!("data: {chunk}\n\n"));
    }

    expected.push_str("data: [DONE]\n\n");

    expected
}

fn tool_call_fixture() -> Vec<WireToolCall> {
    vec![
        WireToolCall {
            id: "call_1".to_owned(),
            name: "get_weather".to_owned(),
            arguments: json!({"city":"Paris"}).to_string(),
        },
        WireToolCall {
            id: "call_2".to_owned(),
            name: "set_timer".to_owned(),
            arguments: "{}".to_owned(),
        },
    ]
}

#[test]
fn chat_stream_emits_tool_calls_chunk_then_terminal_reason() {
    let calls = tool_call_fixture();
    let usage = json!({"prompt_tokens":3,"completion_tokens":2,"total_tokens":5});
    let mut output: Vec<u8> = Vec::new();
    let mut response =
        Response::for_writer(&mut output, ApiKind::Chat, "chatcmpl-tc", 123, true, false);

    response.start().unwrap();
    response
        .finish("", Some(&calls), "tool_calls", usage)
        .unwrap();

    let wire = String::from_utf8(output).unwrap();
    let (headers, body) = wire.split_once("\r\n\r\n").unwrap();

    assert!(headers.contains("Content-Type: text/event-stream"));

    // One indexed tool_calls delta chunk before the empty-delta finish chunk.
    let role_chunk =
        json!({"index":0,"finish_reason":null,"delta":{"content":"","role":"assistant"}});
    let call_chunk = json!({
        "index":0,
        "finish_reason":null,
        "delta":{"tool_calls":[
            json!({"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"},"index":0}),
            json!({"id":"call_2","type":"function","function":{"name":"set_timer","arguments":"{}"},"index":1}),
        ]}
    });
    let finish_chunk = json!({"index":0,"finish_reason":"tool_calls","delta":{}});

    let expected = expected_stream(
        "chatcmpl-tc",
        "chat.completion.chunk",
        &[role_chunk, call_chunk, finish_chunk],
        None,
    );

    assert_eq!(body, &expected);
}

#[test]
fn chat_object_nulls_content_when_only_tool_calls_are_present() {
    let calls = tool_call_fixture();
    let usage = json!({"prompt_tokens":3,"completion_tokens":2,"total_tokens":5});
    let mut output: Vec<u8> = Vec::new();
    let mut response = Response::for_writer(
        &mut output,
        ApiKind::Chat,
        "chatcmpl-tco",
        123,
        false,
        false,
    );

    response.start().unwrap();
    response
        .finish("", Some(&calls), "tool_calls", usage)
        .unwrap();

    let wire = String::from_utf8(output).unwrap();
    let (_headers, body) = wire.split_once("\r\n\r\n").unwrap();
    let parsed: Value = serde_json::from_str(body).unwrap();
    let message = &parsed["choices"][0]["message"];

    // Empty visible text with structured calls: content becomes null and the
    // calls move into the message.
    assert_eq!(message["content"], Value::Null);
    assert_eq!(parsed["choices"][0]["finish_reason"], json!("tool_calls"));

    let tool_calls = message["tool_calls"].as_array().unwrap();

    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0]["id"], json!("call_1"));
    assert_eq!(tool_calls[0]["type"], json!("function"));
    assert_eq!(tool_calls[0]["function"]["name"], json!("get_weather"));
    assert_eq!(
        tool_calls[0]["function"]["arguments"],
        json!("{\"city\":\"Paris\"}")
    );
}

#[test]
fn chat_object_keeps_string_content_without_tool_calls() {
    let mut output: Vec<u8> = Vec::new();
    let mut response = Response::for_writer(
        &mut output,
        ApiKind::Chat,
        "chatcmpl-tcp",
        123,
        false,
        false,
    );

    response.start().unwrap();
    response
        .finish(
            "",
            None,
            "stop",
            json!({"prompt_tokens":3,"completion_tokens":0,"total_tokens":3}),
        )
        .unwrap();

    let wire = String::from_utf8(output).unwrap();
    let (_headers, body) = wire.split_once("\r\n\r\n").unwrap();
    let parsed: Value = serde_json::from_str(body).unwrap();
    let message = &parsed["choices"][0]["message"];

    assert_eq!(message["content"], json!(""));
    assert!(message.get("tool_calls").is_none());
}
