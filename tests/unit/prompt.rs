use super::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct References {
    template_sha256: String,
    tokenizer_sha256: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    context: Value,
    text: Option<String>,
    #[serde(default)]
    token_ids: Vec<u32>,
    error: Option<String>,
}

fn references() -> References {
    serde_json::from_str(include_str!("../fixtures/prompt/references.json"))
        .expect("Transformers reference fixtures")
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn checkpoint_template_matches_transformers_reference_bytes_and_errors() {
    let references = references();
    let template = fixture_template();
    let source = include_bytes!("../fixtures/prompt/chat_template.jinja");

    assert_eq!(sha256(source), references.template_sha256);

    for case in references.cases {
        let result = template.render_context(&case.context);

        if let Some(expected) = case.error {
            let error = result.expect_err(&case.id);

            assert!(
                format!("{error:#}").contains(&expected),
                "{}: {error:#}",
                case.id
            );

            continue;
        }

        assert_eq!(result.unwrap(), case.text.unwrap(), "{}", case.id);
    }
}

#[test]
fn renderer_matches_reference_token_ids_with_the_checkpoint_tokenizer() {
    let Some(model) = std::env::var_os("CHERENKOV_MODEL_DIR") else {
        eprintln!("set CHERENKOV_MODEL_DIR to verify reference token IDs");

        return;
    };
    let path = Path::new(&model).join("tokenizer.json");
    let references = references();
    let bytes = std::fs::read(&path).unwrap();

    assert_eq!(sha256(&bytes), references.tokenizer_sha256);

    let tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
    let template = fixture_template();

    for case in references
        .cases
        .into_iter()
        .filter(|case| case.error.is_none())
    {
        let rendered = template.render_context(&case.context).unwrap();
        let encoded = tokenizer.encode(rendered, false).unwrap();

        assert_eq!(encoded.get_ids(), case.token_ids, "{}", case.id);
    }
}

#[test]
fn chat_adapter_preserves_reference_bytes_and_safe_unicode_prefixes() {
    let template = fixture_template();

    for case in references()
        .cases
        .into_iter()
        .filter(|case| case.error.is_none())
    {
        if case.context["enable_thinking"] != false
            || case.context["add_generation_prompt"] != true
            || case.context.get("preserve_thinking").is_some()
        {
            continue;
        }

        let messages = case.context["messages"].as_array().unwrap();
        let prompt = template.chat(messages, None).unwrap();

        assert_eq!(Some(&prompt.text), case.text.as_ref(), "{}", case.id);

        for boundary in prompt.boundaries {
            assert!(prompt.text.is_char_boundary(boundary), "{}", case.id);
        }

        assert!(prompt.boundaries[0] <= prompt.boundaries[1]);
    }
}

#[test]
fn cli_and_chat_share_the_loaded_template_and_developer_alias() {
    let template = fixture_template();
    let cli = template.user("  Describe a café.  ").unwrap();
    let chat = template
        .chat(
            &[json!({"role":"user", "content":"  Describe a café.  "})],
            None,
        )
        .unwrap();

    assert_eq!(cli.text, chat.text);
    assert_eq!(cli.boundaries, chat.boundaries);

    let mut messages = vec![
        json!({"role":"developer", "content":"Be brief."}),
        json!({"role":"user", "content":"Hi"}),
    ];
    let developer = template.chat(&messages, None).unwrap();
    messages[0]["role"] = json!("system");

    assert_eq!(developer.text, template.chat(&messages, None).unwrap().text);
}

#[test]
fn loading_prefers_the_jinja_file_and_supports_embedded_templates() {
    let directory = tempfile::tempdir().unwrap();

    assert!(ChatTemplate::load(directory.path()).unwrap().is_none());

    let config = directory.path().join("tokenizer_config.json");

    for template in [
        json!("embedded {{ messages[0].content }}"),
        json!([{"name":"default", "template":"embedded {{ messages[0].content }}"}]),
    ] {
        std::fs::write(&config, json!({"chat_template":template}).to_string()).unwrap();

        let loaded = ChatTemplate::load(directory.path()).unwrap().unwrap();

        assert_eq!(loaded.user("Hello").unwrap().text, "embedded Hello");
    }

    std::fs::write(
        directory.path().join("chat_template.jinja"),
        "file {{ messages[0].content }}",
    )
    .unwrap();

    let loaded = ChatTemplate::load(directory.path()).unwrap().unwrap();

    assert_eq!(loaded.user("Hello").unwrap().text, "file Hello");
    std::fs::write(
        directory.path().join("chat_template.jinja"),
        "{% invalid %}",
    )
    .unwrap();
    assert!(ChatTemplate::load(directory.path()).is_err());
}

#[test]
fn raw_prompt_has_no_template_or_cache_boundaries() {
    let prompt = Prompt::raw("plain text".into());

    assert_eq!(prompt.text, "plain text");
    assert_eq!(prompt.boundaries, [0, 0]);
}

fn weather_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            },
            "strict": false
        }
    })
}

#[test]
fn typed_text_content_parts_render_like_string_content() {
    let template = fixture_template();

    let messages = |content: Value| {
        vec![
            json!({"role": "system", "content": "Be brief."}),
            json!({"role": "user", "content": content}),
        ]
    };

    let typed = template
        .chat(&messages(json!([{"type": "text", "text": "Hi"}])), None)
        .unwrap();
    let plain = template.chat(&messages(json!("Hi")), None).unwrap();

    assert_eq!(typed.text, plain.text);
    assert_eq!(typed.boundaries, plain.boundaries);

    let multi = template
        .chat(
            &messages(json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])),
            None,
        )
        .unwrap();

    assert_eq!(
        multi.text,
        template.chat(&messages(json!("ab")), None).unwrap().text
    );

    // The template renders typed parts; shapes it cannot honor are rejected
    // before rendering: untyped string parts, text parts without text, media
    // parts, and non-object parts.
    for content in [
        json!(["hello"]),
        json!([{"type": "text"}]),
        json!([{"type": "image_url", "image_url": {"url": "x"}}]),
        json!([1]),
    ] {
        assert!(
            template.chat(&messages(content.clone()), None).is_err(),
            "{content}"
        );
    }
}

#[test]
fn chat_tools_render_the_tool_block() {
    let template = fixture_template();
    let before = template
        .chat(&[json!({"role":"user","content":"Hi"})], None)
        .unwrap();
    let after = template
        .chat(
            &[json!({"role":"user","content":"Hi"})],
            Some(&[weather_tool()]),
        )
        .unwrap();

    assert!(!before.text.contains("# Tools"));
    assert!(after.text.contains("# Tools"));
    assert!(after.text.contains("<tools>"));
    assert!(after.text.contains("\"name\":\"get_weather\""));
    assert!(after.text.contains("required"));
    assert!(
        after
            .text
            .contains("If you choose to call a function ONLY reply in the following format")
    );
}

#[test]
fn chat_history_renders_assistant_tool_calls_and_tool_responses() {
    let template = fixture_template();
    let messages = vec![
        json!({"role":"user","content":"Weather in Paris?"}),
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name":"get_weather","arguments":{"city":"Paris"}}
            }]
        }),
        json!({"role":"tool","tool_call_id":"call_1","content":"Sunny, 24C"}),
        json!({"role":"user","content":"And tomorrow?"}),
    ];
    let prompt = template.chat(&messages, Some(&[weather_tool()])).unwrap();

    assert!(prompt.text.contains("<tool_call>\n<function=get_weather>"));
    assert!(
        prompt
            .text
            .contains("<parameter=city>\nParis\n</parameter>")
    );
    assert!(prompt.text.contains("</function>\n</tool_call>"));
    assert!(
        prompt
            .text
            .contains("<tool_response>\nSunny, 24C\n</tool_response>")
    );
}

#[test]
fn chat_history_rejects_malformed_tool_calls() {
    let template = fixture_template();
    let ok = |args: Value| {
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name":"get_weather","arguments":args}
            }]
        })
    };

    // Well-formed history is accepted.
    let good = template
        .chat(
            &[
                json!({"role":"user","content":"Hi"}),
                ok(Value::String(json!({"city":"Paris"}).to_string())),
            ],
            None,
        )
        .unwrap();

    assert!(good.text.contains("<tool_call>\n<function=get_weather>"));

    // The `arguments` wire form must be a string that decodes to an object.
    for bad_args in [json!(1), json!("[1,2]"), json!("not json")] {
        let messages = vec![json!({"role":"user","content":"Hi"}), ok(bad_args.clone())];

        assert!(template.chat(&messages, None).is_err(), "{bad_args}");
    }

    // Retained session history carries the decoded object form, which is
    // accepted on the turn-back path.
    let history = vec![
        json!({"role":"user","content":"Hi"}),
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name":"get_weather","arguments":{"city":"Paris"}}
            }]
        }),
    ];

    assert!(template.chat(&history, None).is_ok());
}
