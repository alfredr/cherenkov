use super::*;
use crate::{
    config::Config,
    server::tests::fixtures::{begin_turn, message, new_session},
};

#[test]
fn session_input_keeps_history_and_settings_without_rewriting_json() {
    let (store, id) = new_session(
        &Config::default(),
        json!({"temperature": 0.7, "top_k": 20, "context_tokens": 64}),
    );
    let first = message(&id, "Hello");

    begin_turn(&store, &first, "first")
        .prepare("Hi", None, Default::default(), None)
        .unwrap()
        .publish();

    let mut body = message(&id, "Continue");
    body["top_k"] = json!(3);
    let original = body.clone();
    let turn = begin_turn(&store, &body, "second");
    let defaults = Defaults {
        sampling: Sampling {
            temperature: 1.2,
            seed: Some(999),
            ..Sampling::default()
        },
        ..Defaults::default()
    };
    let request = parse_request(&body, ApiKind::Chat, &defaults, Some(&turn.input)).unwrap();

    assert_eq!(body, original);
    assert_eq!(request.sampling.temperature, 0.7);
    assert_eq!(request.sampling.top_k, 3);
    assert_eq!(request.sampling.seed, None);
    assert_eq!(request.context, Some(64));
    assert_eq!(
        request.prompt.text,
        concat!(
            "<|im_start|>user\nHello<|im_end|>\n",
            "<|im_start|>assistant\n<think>\n\n</think>\n\nHi<|im_end|>\n",
            "<|im_start|>user\nContinue<|im_end|>\n",
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
    );
}

#[test]
fn request_values_override_reloadable_defaults() {
    let defaults = Defaults {
        max_tokens: 128,
        stream: true,
        include_usage: true,
        no_eos: false,
        sampling: crate::sampling::Sampling::default(),
    };
    let r = parse_request(
        &json!({"prompt":"hello"}),
        ApiKind::Completion,
        &defaults,
        None,
    )
    .unwrap();

    assert_eq!(r.max_tokens, 128);
    assert!(r.stream && r.include_usage);

    let r = parse_request(
        &json!({
            "prompt": "hello",
            "max_tokens": 7,
            "stream": false,
            "stream_options": {"include_usage": false},
        }),
        ApiKind::Completion,
        &defaults,
        None,
    )
    .unwrap();

    assert_eq!(r.max_tokens, 7);
    assert!(!r.stream && !r.include_usage);
}

#[test]
fn chat_template_preserves_roles_and_defaults() {
    let body = json!({
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_completion_tokens": 17,
    });
    let r = parse_request(&body, ApiKind::Chat, &Defaults::default(), None).unwrap();

    assert_eq!(
        r.prompt.text,
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert_eq!(r.max_tokens, 17);
    assert!(r.stream && r.include_usage);
    assert_eq!(
        parse_request(
            &json!({"prompt":"raw"}),
            ApiKind::Completion,
            &Defaults::default(),
            None,
        )
        .unwrap()
        .prompt
        .text,
        "raw"
    );
}

#[test]
fn chat_request_accepts_typed_text_content_parts() {
    let body = json!({
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        ],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_completion_tokens": 17,
    });
    let r = parse_request(&body, ApiKind::Chat, &Defaults::default(), None).unwrap();

    assert_eq!(
        r.prompt.text,
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}

#[test]
fn reject_unsupported_generation_instead_of_ignoring_it() {
    for extra in [
        json!({"temperature":-0.7}),
        json!({"n":2}),
        json!({"top_p":0}),
        json!({"top_k":-1}),
        json!({"seed":-1}),
        json!({"presence_penalty":3}),
        json!({"reasoning_effort":"high"}),
        json!({"previous_response_id":"other"}),
        json!({"max_tokens":0}),
        json!({"max_tokens":-1}),
        json!({"stream":"yes"}),
        json!({"tools":[]}),
        json!({"stop":["x"]}),
        json!({"model":"other"}),
        json!({"response_format":{"type":"json_object"}}),
    ] {
        let mut request = json!({"prompt":"test"});

        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert!(
            parse_request(&request, ApiKind::Completion, &Defaults::default(), None).is_err(),
            "{request}"
        );
    }

    assert!(
        parse_request(
            &json!({"messages":[]}),
            ApiKind::Chat,
            &Defaults::default(),
            None
        )
        .is_err()
    );
    assert!(
        parse_request(
            &json!({"messages":[{"role":"user","content":[{"type":"image_url"}]}]}),
            ApiKind::Chat,
            &Defaults::default(),
            None,
        )
        .is_err()
    );
    assert!(
        parse_request(
            &json!({"messages":[{"role":"user","content":["hello"]}]}),
            ApiKind::Chat,
            &Defaults::default(),
            None,
        )
        .is_err()
    );
    assert!(
        parse_request(
            &json!({"messages":[{"role":"user","content":[{"type":"text"}]}]}),
            ApiKind::Chat,
            &Defaults::default(),
            None,
        )
        .is_err()
    );
}

fn preparation_tokenizer() -> ChatTokenizer {
    use tokenizers::{
        Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };

    let model = WordLevel::builder()
        .vocab(
            [("[UNK]", 0), ("hello", 1), ("world", 2), ("last", 3)]
                .into_iter()
                .map(|(word, id)| (word.to_owned(), id))
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut inner = Tokenizer::new(model);

    inner.with_pre_tokenizer(Some(Whitespace));

    ChatTokenizer {
        inner,
        template: Some(crate::prompt::fixture_template()),
        im_end: 4,
        endoftext: 5,
    }
}

#[test]
fn preparation_keeps_validated_tokens_prefix_boundaries_and_eos_policy() {
    let tok = preparation_tokenizer();
    let options = Options {
        max_ctx: 7,
        ..Options::default()
    };
    let mut request = parse_request(
        &json!({"prompt":"hello world last", "max_tokens":2}),
        ApiKind::Completion,
        &Defaults {
            no_eos: true,
            ..Defaults::default()
        },
        None,
    )
    .unwrap();
    // The incomplete word retokenizes differently: only the first token matches.
    request.prompt.boundaries = ["hello wor".len(), "hello world".len()];
    let prepared = request.prepare(&tok, &options, 2).unwrap();

    assert_eq!(prepared.ids, vec![1, 2, 3]);
    // One matching token is held back for MTP's following-token dependency.
    assert_eq!(prepared.boundaries, vec![0, 2]);
    assert_eq!(prepared.request.max_tokens, 2);
    assert!(prepared.options.no_eos);
    assert!(!options.no_eos);
}

#[test]
fn preparation_rejects_empty_prompts_and_output_or_context_overflow() {
    let tok = preparation_tokenizer();

    for (prompt, output_limit, context, expected) in [
        ("", 2, 7, "at least one token"),
        ("hello world last", 1, 7, "output policy"),
        ("hello world last", 2, 6, "exceeding --max-ctx"),
    ] {
        let request = parse_request(
            &json!({"prompt":prompt, "max_tokens":2, "stream":true}),
            ApiKind::Completion,
            &Defaults::default(),
            None,
        )
        .unwrap();
        let options = Options {
            max_ctx: context,
            ..Options::default()
        };
        let error = request.prepare(&tok, &options, output_limit).err().unwrap();

        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn shared_chat_prefix_keeps_mtps_following_token_stable() {
    let Some(model) = std::env::var_os("CHERENKOV_MODEL_DIR") else {
        eprintln!("set CHERENKOV_MODEL_DIR to verify chat prefix token boundaries");

        return;
    };
    let tok = ChatTokenizer::load(std::path::Path::new(&model)).unwrap();

    for drafts in [0, 2] {
        let options = Options {
            drafts,
            ..Options::default()
        };
        let prepared: Vec<_> = ["What is a hash table?", "Where are hash tables used?"]
            .into_iter()
            .map(|content| {
                let body = json!({
                    "messages": [
                        {"role": "system", "content": "Explain things clearly."},
                        {"role": "user", "content": content},
                    ],
                    "max_tokens": 2,
                });

                super::parse_request(
                    &body,
                    ApiKind::Chat,
                    &Defaults::default(),
                    None,
                    tok.template.as_ref(),
                )
                .unwrap()
                .prepare(&tok, &options, 2)
                .unwrap()
            })
            .collect();
        let end = prepared[0].boundaries[0];

        assert!(end > 0);
        assert_eq!(end, prepared[1].boundaries[0]);

        let following = usize::from(drafts > 0);

        assert_eq!(
            prepared[0].ids[..end + following],
            prepared[1].ids[..end + following]
        );
    }
}

#[test]
fn chat_tools_render_the_tools_block_and_build_a_contract() {
    let body = json!({
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                }
            }
        }],
    });
    let request = parse_request(&body, ApiKind::Chat, &Defaults::default(), None).unwrap();

    let text = &request.prompt.text;

    assert!(text.contains("# Tools"), "{text}");
    assert!(text.contains("<tools>") && text.contains("</tools>"));
    assert!(text.contains("\"name\":\"get_weather\""));
    // The contract-facing shape normalizes the flag the model cannot guarantee.
    assert!(text.contains("\"strict\":false"));
    assert!(request.tool_contract.is_some());
}

#[test]
fn chat_tool_choice_none_suppresses_the_tools_block() {
    let body = json!({
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather"}
        }],
        "tool_choice": "none",
    });
    let request = parse_request(&body, ApiKind::Chat, &Defaults::default(), None).unwrap();

    assert!(!request.prompt.text.contains("# Tools"));
    assert!(request.tool_contract.is_none());
}

#[test]
fn chat_tools_reject_unsupported_controls() {
    let fn_call = |name: &str| {
        json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": name}}],
        })
    };

    // Bad name grammar, missing function object, wrong tool type, strict flag,
    // and an explicit function choice are all rejected while tools are present.
    for extra in [
        json!({"tools": [{"type": "function", "function": {"name": "bad name"}}]}),
        json!({"tools": [{"type": "function", "function": {}}]}),
        json!({"tools": [{"type": "retrieval", "function": {"name": "get_weather"}}]}),
        json!({"tools": [{"type": "function", "function": {"name": "x", "strict": true}}]}),
        json!({"tool_choice": "required"}),
        json!({"tool_choice": {"type": "function", "function": {"name": "x"}}}),
        json!({"parallel_tool_calls": false}),
    ] {
        let mut body = fn_call("get_weather").as_object().unwrap().clone();

        body.extend(extra.as_object().unwrap().clone());

        let request = Value::Object(body);

        assert!(
            parse_request(&request, ApiKind::Chat, &Defaults::default(), None).is_err(),
            "{request:?}"
        );
    }

    // `tool_choice: "none"` is the one tool_choice that drops the tools block.
    let mut body = fn_call("get_weather").as_object().unwrap().clone();

    body.insert("tool_choice".to_owned(), json!("none"));

    let request = parse_request(
        &Value::Object(body),
        ApiKind::Chat,
        &Defaults::default(),
        None,
    )
    .unwrap();

    assert!(request.tool_contract.is_none());

    // Legacy controls are still rejected.
    for legacy in [
        json!({"functions": [{"name": "x"}]}),
        json!({"function_call": {"name": "x"}}),
    ] {
        let mut body = fn_call("get_weather").as_object().unwrap().clone();

        body.extend(legacy.as_object().unwrap().clone());

        let request = Value::Object(body);

        assert!(
            parse_request(&request, ApiKind::Chat, &Defaults::default(), None).is_err(),
            "{request:?}"
        );
    }
}

fn parse_request(
    body: &Value,
    kind: ApiKind,
    defaults: &Defaults,
    session: Option<&SessionInput>,
) -> Result<Request> {
    super::parse_request(
        body,
        kind,
        defaults,
        session,
        Some(&crate::prompt::fixture_template()),
    )
}
