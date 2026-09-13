# Running Cherenkov

See the [README](../README.md) for installation and CLI examples.

## HTTP server

```sh
cherenkov serve /path/to/model --port 8080 --max-ctx 4096
```

Connect clients to `http://127.0.0.1:8080/v1` with model `cherenkov`.
The server binds to localhost and does not authenticate HTTP requests.
Clients that require an API key can use a placeholder.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Readiness |
| `GET /v1/models` | Model list |
| `POST /v1/chat/completions` | Text chat |
| `POST /v1/completions` | String-prompt completion |

Both completion endpoints support JSON and SSE, finish reasons, token usage,
and `stream_options.include_usage`. Requests must use Content-Length;
chunked request bodies are unsupported.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"cherenkov","messages":[{"role":"user","content":"Explain hash collisions."}],"max_tokens":128,"stream":true}'
```

### Generation options

| Field | Accepted values | Default |
| --- | --- | --- |
| `temperature` | 0–2; 0 is greedy | 0 |
| `top_p` | Greater than 0, at most 1 | 1 |
| `top_k` | Nonnegative integer; 0 keeps all tokens | 0 |
| `seed` | Unsigned integer | Random |
| `presence_penalty`, `frequency_penalty` | −2–2 | 0 |
| `n` | 1 | 1 |

These defaults can be set in [TOML](server-config.md). Penalties count prompt
and output tokens. Selection applies penalties, temperature, top-k, then top-p.
Sampling or penalties disable MTP verification. CLI sampling uses the matching
hyphenated flags, such as `--top-k` and `--presence-penalty`.

Both `max_tokens` and `max_completion_tokens` are accepted. Stop strings,
JSON schemas, images, reasoning effort, template controls, and the
Responses API are unsupported. Unsupported generation controls return an error.
`--raw`, `--check`, and `--repeat` are CLI-only.

### Tool calling

Chat also accepts `tools` and `tool_choice`. Tools must be function
definitions (`type` `function`, a name matching `[A-Za-z0-9_-]{1,64}`, and an
object `parameters` schema); `strict: true` and non-object schemas are
rejected. `tool_choice` accepts `auto` (default) and `none`; `required` and a
specific-function choice are rejected because the checkpoint cannot be
obligated to call a function. `parallel_tool_calls` is accepted only when it
is `true` while tools are enabled or absent.

The checkpoint renders the tools block and emits calls as XML-style markup.
The server parses that markup incrementally and holds back bytes that could
belong to a terminal marker. Text before the calls is preserved as `content`,
with separator whitespace immediately before the marker removed. Parsed call
markup is excluded from content in streaming responses, JSON responses, and
retained session history. A call-only assistant message has `content: null`
in JSON responses and session history; its stream contains no nonempty
content delta. Streaming returns the calls together in an indexed
`tool_calls` delta before the final finish-reason chunk.

Calls receive `finish_reason: "tool_calls"` when the model stops naturally.
A token-budget cutoff keeps `finish_reason: "length"`, even if calls were
recovered. Recovery can retain a call whose name and parameters are complete
but closing tags are missing, or keep earlier calls and discard a malformed
or explanatory suffix. If the region cannot be recovered, its bytes are
returned verbatim as content, including in streaming responses. Discarded
suffixes and arguments that violate a declared parameter type produce a
one-line diagnostic.

The client executes the calls and sends `role: "tool"` messages with the
results. Retained sessions keep structured calls for the next turn without
duplicating their markup in content. Session history stores decoded argument
objects; API responses and fresh requests carry `arguments` as a JSON string.

### Chat template

Chat accepts text messages with system, developer, user, assistant, and tool roles.
Developer messages are mapped to system messages. A system message must come
first, and the conversation must contain a user query.

The renderer loads `chat_template.jinja`, falling back to the string or named
`default` template in `tokenizer_config.json`. It supplies `messages`,
`add_generation_prompt=true`, and `enable_thinking=false`. The template controls
formatting and message validation. Raw CLI prompts and text completions bypass it.

Preparation copies template metadata into the artifact. Re-running `prepare`
fills missing files from an available local or retained source, preserving
existing metadata. Repairs publish a new artifact; active readers keep their
original files.

## Sessions

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/sessions` | Create a session with optional sampling settings and `context_tokens` |
| `GET /v1/sessions/{id}` | Read committed messages, settings, usage, and active request ID |
| `DELETE /v1/sessions/{id}` | Delete an idle session |
| `GET /v1/requests` | List registered requests and cancellation flags |
| `POST /v1/requests/{id}/cancel` | Cancel a queued or active request |

```sh
curl http://127.0.0.1:8080/v1/sessions \
  -H 'Content-Type: application/json' \
  -d '{"temperature":0.7,"top_k":20,"seed":42,"context_tokens":2048}'

# Send only new messages when using a session ID.
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"session-...","request_id":"turn-1","messages":[{"role":"user","content":"Explain hash tables."}],"max_tokens":128,"stream":true}'

curl -X POST http://127.0.0.1:8080/v1/requests/turn-1/cancel
```

Without `session_id`, send the full conversation. A session permits one in-flight
turn. Different sessions progress in turn on the shared model.

Successful turns save messages, effective sampling settings, and RNG state.
Omitted sampling fields inherit session settings. An explicit seed restarts
the random stream; greedy turns consume no random draws. Sessions are held
in RAM and are subject to [retention limits](server-config.md).
See [statistics](stats.md#requests-sessions-and-memory) for session usage fields.

Cancellation discards the new turn, including its settings and RNG changes.
Retry from the last committed history. Streamed text is provisional until
completion. The current GPU step or prefill chunk finishes before cancellation
takes effect. Cancellation cannot undo a turn once final publication commits it.
Disconnects detected during writes and full output queues cancel unfinished work.

Active cancellation returns `finish_reason: "cancelled"`. Cancellation before
admission returns HTTP 499. Missing sessions return 404; busy sessions and
duplicate request IDs return 409; exhausted request capacity returns 503.

A supplied `request_id` must contain 1–80 ASCII letters, digits, hyphens, or
underscores and be unique among registered requests. Otherwise the server
assigns one. It appears in completion IDs and the SSE `X-Request-ID` header.

`context_tokens` limits prompt, output, and speculative headroom within the
server's capacity. Lower limits reduce checkpoint reservations but do not
resize the shared GPU buffers.

## Prefix cache

The server restores the longest cached prompt prefix and processes the rest.
Checkpoints contain attention, recurrent, convolution, PLE, token, and MTP
state. They store model state, not responses, and disappear on server exit.

`--prefix-cache-mb` defaults to 512 MiB; 0 disables the cache. TOML also sets
entry and idle limits. The cache evicts least recently used entries and skips
oversized checkpoints. A short checkpoint uses about 113 MiB plus context
storage, so the default budget holds roughly four short prefixes.

Checkpoints are saved at message boundaries, complete prompts, and selected
prefill chunks. MTP's following-token dependency is checked before reuse.
Cached and fresh runs can choose different tokens on close argmax decisions
because expert accumulation order can differ.

Reused tokens appear in `usage.prompt_tokens_details.cached_tokens`. For SSE,
set `stream_options.include_usage` to receive usage in the final event.

## Precision and memory

The default is 4-bit experts, two adaptive MTP drafts, and no deadline cut.
`--drafts 0` disables speculation and omits the draft head.

`--experts 3` or `--experts 2` applies to routed experts in prefill and decode.
Mixed mode requires 4-bit residents and uses `--miss-experts` for fetched
misses. Shared experts and dense weights keep their original precision.
Lower precision changes output. A nonzero `--cut-weak` can skip late weak
experts, making output depend on disk timing.

See [storage](storage.md) for packing low-bit stores. Context uses about
22.5 KB per token, reducing the adaptive expert pool. `--pool-gb N` sets only
that pool, in decimal GB; `--pool-gb max` uses more of the device's budget.
The runner checks prompt, requested output, and draft headroom against
`--max-ctx` before loading the GPU model.

## Timing

The CLI writes generated text to stdout and load, prefill, and decode timings
to stderr. `--max-tokens N --no-eos --repeat N` measures repeated generation
with loaded weights and fresh sequence state. `--check` runs the slow CPU
reference and must not be used for speed measurements.

Warm repeats retain the expert pool and can change accumulation order. Use
fresh processes with matching options for output comparisons. See the
[benchmark suite](../benchmarks/README.md) for measured workloads.
