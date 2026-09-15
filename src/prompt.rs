//! Checkpoint-owned Jinja formatting, compiled once when the tokenizer loads.

use anyhow::{Context, Result, ensure};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{Map, Value, json};
use std::{io::ErrorKind as IoErrorKind, path::Path};

pub(crate) struct Prompt {
    pub text: String,
    /// Safe byte prefixes, checked against the complete rendered prompt.
    pub boundaries: [usize; 2],
}

impl Prompt {
    pub(crate) fn raw(text: String) -> Self {
        Self {
            text,
            boundaries: [0, 0],
        }
    }
}

pub(crate) struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub(crate) fn load(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join("chat_template.jinja");
        let source = match std::fs::read_to_string(&path) {
            Ok(source) => Some(source),
            Err(error) if error.kind() == IoErrorKind::NotFound => embedded_template(model_dir)?,
            Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
        };

        source.map(Self::new).transpose()
    }

    fn new(source: String) -> Result<Self> {
        let mut env = Environment::new();

        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_recursion_limit(64);
        env.set_fuel(Some(1_000_000));
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_template_owned("chat", source)
            .context("compiling checkpoint chat template")?;

        Ok(Self { env })
    }

    pub(crate) fn user(&self, content: &str) -> Result<Prompt> {
        self.chat(&[json!({"role": "user", "content": content})], None)
    }

    /// Render a chat conversation; `tools` are the shaped function definitions
    /// for the template's tools system block (omitted when empty).
    pub(crate) fn chat(&self, messages: &[Value], tools: Option<&[Value]>) -> Result<Prompt> {
        let mut messages = text_messages(messages)?;
        let tools = tools.filter(|t| !t.is_empty());
        let text = self.render(&messages, true, tools)?;
        let message_end = common_prefix(&text, &self.render(&messages, false, tools)?);
        // Prefix-only renders can be invalid (the template requires a user query).
        // Keep the last user with empty content as a cache probe, and trust only
        // bytes that also occur at the beginning of the full rendered prompt.
        let mut stable_end = 0;

        if let Some(last_user) = messages.iter().rposition(|m| m["role"] == "user") {
            messages.truncate(last_user + 1);

            messages[last_user]["content"] = json!("");

            if let Ok(prefix) = self.render(&messages, false, tools) {
                stable_end = common_prefix(&text, &prefix);
            }
        }

        Ok(Prompt {
            text,
            boundaries: [stable_end, message_end],
        })
    }

    fn render(
        &self,
        messages: &[Value],
        add_generation_prompt: bool,
        tools: Option<&[Value]>,
    ) -> Result<String> {
        // Preserve the engine's direct-answer default. Other formatting and
        // reasoning-history defaults come from the checkpoint's template.
        let mut context = json!({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": false,
        });

        // The template renders a tools system block only for a non-empty list.
        if let Some(tools) = tools {
            context["tools"] = json!(tools.to_vec());
        }

        self.render_context(&context)
    }

    fn render_context(&self, context: &Value) -> Result<String> {
        self.env
            .get_template("chat")?
            .render(context)
            .map_err(|error| anyhow::anyhow!("chat template: {error}"))
    }
}

fn embedded_template(model_dir: &Path) -> Result<Option<String>> {
    let path = model_dir.join("tokenizer_config.json");
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == IoErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    let config: Value =
        serde_json::from_slice(&bytes).context("reading tokenizer configuration")?;
    let template = &config["chat_template"];

    if let Some(source) = template.as_str() {
        return Ok(Some(source.to_owned()));
    }

    // Transformers also saves named templates; chat without tools uses "default".
    Ok(template.as_array().and_then(|templates| {
        templates
            .iter()
            .find(|t| t["name"] == "default")?
            .get("template")?
            .as_str()
            .map(str::to_owned)
    }))
}

/// Names accepted on the wire; the tokenizer side uses a longer bound for
/// model-emitted names.
pub(crate) fn valid_function_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Arrays pass through to the checkpoint template, whose content macro
/// renders the typed parts; only part shapes the template cannot honor are
/// rejected here.
fn validate_content(message: &Value) -> Result<()> {
    for part in message["content"].as_array().into_iter().flatten() {
        let obj = if part.is_string() {
            anyhow::bail!("content parts must be typed objects")
        } else {
            part.as_object()
                .ok_or_else(|| anyhow::anyhow!("content parts must be typed objects"))?
        };
        let typ = obj.get("type").and_then(|v| v.as_str());

        match typ {
            Some("text") => {
                obj.get("text")
                    .and_then(|v| v.as_str())
                    .context("text content part requires a string 'text'")?;
            }
            Some("image") | Some("image_url") => {
                anyhow::bail!("image input is not supported")
            }
            Some("video") => anyhow::bail!("video input is not supported"),
            Some(other) => anyhow::bail!("content part type '{other}' is not supported"),
            None => anyhow::bail!("content parts must be typed objects"),
        }
    }

    Ok(())
}

fn text_messages(messages: &[Value]) -> Result<Vec<Value>> {
    ensure!(!messages.is_empty(), "messages must not be empty");

    let mut messages = messages.to_vec();

    for (index, message) in messages.iter_mut().enumerate() {
        validate_content(message)?;
        let role = message["role"]
            .as_str()
            .context("message role must be a string")?
            .to_owned();

        ensure!(
            ["system", "developer", "user", "assistant", "tool"].contains(&role.as_str()),
            "unsupported role {role}"
        );

        match role.as_str() {
            "assistant" => validate_assistant_message(message, index)?,
            "tool" => validate_tool_message(message, index)?,
            _ => validate_plain_message(message)?,
        }

        if role == "developer" {
            message["role"] = json!("system");
        }
    }

    Ok(messages)
}

fn validate_plain_message(message: &Value) -> Result<()> {
    ensure!(
        message["content"].is_string() || message["content"].is_array(),
        "message content must be text"
    );
    ensure!(
        message["tool_calls"].is_null() || message["tool_calls"] == json!([]),
        "tool_calls are only valid on assistant messages"
    );
    ensure!(
        message["function_call"].is_null(),
        "function_call is only valid on assistant messages"
    );
    ensure!(
        message["tool_call_id"].is_null(),
        "tool_call_id is only valid on tool messages"
    );
    ensure_reasoning_content_absent(message)?;

    Ok(())
}

fn validate_tool_message(message: &Value, index: usize) -> Result<()> {
    ensure!(
        message["content"].is_string() || message["content"].is_array(),
        "tool message {index} content must be text"
    );
    ensure!(
        message["tool_calls"].is_null() || message["tool_calls"] == json!([]),
        "tool messages cannot contain tool_calls"
    );
    ensure!(
        message["function_call"].is_null(),
        "function_call is only valid on assistant messages"
    );

    if !message["tool_call_id"].is_null() {
        message["tool_call_id"]
            .as_str()
            .context("tool_call_id must be a string")?;
    }

    ensure_reasoning_content_absent(message)?;

    Ok(())
}

fn ensure_reasoning_content_absent(message: &Value) -> Result<()> {
    if !message["reasoning_content"].is_null() {
        let reasoning = message["reasoning_content"]
            .as_str()
            .context("reasoning_content must be a string")?;

        ensure!(
            reasoning.is_empty(),
            "reasoning_content is only valid on assistant messages"
        );
    }

    Ok(())
}

fn validate_assistant_message(message: &mut Value, index: usize) -> Result<()> {
    ensure!(
        message["content"].is_string()
            || message["content"].is_array()
            || message["content"].is_null(),
        "assistant message {index} content must be text or null"
    );
    ensure!(
        message["tool_call_id"].is_null(),
        "message {index} tool_call_id is only valid on tool messages"
    );

    if !message["reasoning_content"].is_null() {
        message["reasoning_content"]
            .as_str()
            .context("assistant reasoning_content must be a string")?;
    }

    // Keep the wire order: a legacy function_call precedes tool_calls.
    let mut calls: Vec<Value> = Vec::new();

    if !message["function_call"].is_null() {
        let legacy = message["function_call"]
            .as_object()
            .context("function_call must be an object")?;
        let (name, arguments) = function_call_name_and_arguments(legacy)?;

        calls.push(tool_call_value(String::new(), name, arguments));
        message.as_object_mut().unwrap().remove("function_call");
    }

    if !message["tool_calls"].is_null() {
        for call in message["tool_calls"]
            .as_array()
            .context("tool_calls must be an array")?
        {
            if !call.is_object() {
                anyhow::bail!("message {index} tool_calls entries must be objects");
            }

            if !call["id"].is_string() {
                anyhow::bail!("message {index} tool_calls entries must contain a string id");
            }

            ensure!(
                call["type"] == json!("function"),
                "only function tool_calls are supported"
            );

            let function = call["function"]
                .as_object()
                .context("tool_calls entries must contain a function object")?;
            let (name, arguments) = function_call_name_and_arguments(function)?;

            calls.push(tool_call_value(
                call["id"].as_str().unwrap().to_owned(),
                name,
                arguments,
            ));
        }
    }

    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }

    Ok(())
}

/// Validate a function reference and decode its `arguments` into the object
/// the checkpoint template iterates. Fresh requests carry the wire form
/// (a JSON string); retained session history carries the decoded object.
fn function_call_name_and_arguments(function: &Map<String, Value>) -> Result<(String, Value)> {
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .context("function name must be a string")?;

    ensure!(
        valid_function_name(name),
        "function name must match [A-Za-z0-9_-]{{1,64}}"
    );

    let arguments = function
        .get("arguments")
        .context("function arguments are required")?;

    let decoded = match arguments {
        Value::String(encoded) if encoded.trim().is_empty() => Value::Object(Map::new()),
        Value::String(encoded) => {
            serde_json::from_str(encoded).context("function arguments must be valid JSON")?
        }
        Value::Object(_) => arguments.clone(),
        other => anyhow::bail!("function arguments must be a JSON string or object, not {other:?}"),
    };

    ensure!(
        decoded.is_object(),
        "function arguments must decode to a JSON object"
    );

    Ok((name.to_owned(), decoded))
}

fn tool_call_value(id: String, name: String, arguments: Value) -> Value {
    json!({
        "id": id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    })
}

fn common_prefix(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(a, b)| a == b)
        .map(|(c, _)| c.len_utf8())
        .sum()
}

#[cfg(test)]
pub(crate) fn fixture_template() -> ChatTemplate {
    ChatTemplate::new(include_str!("../tests/fixtures/prompt/chat_template.jinja").to_owned())
        .expect("checkpoint template fixture")
}

#[cfg(test)]
#[path = "../tests/unit/prompt.rs"]
mod tests;
