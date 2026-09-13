# Serving architecture

See [running](running.md) for the API and [configuration](server-config.md)
for limits and defaults.

## Source map

`src/server.rs` starts the server. Its child modules are:

| Module | Purpose |
| --- | --- |
| `http` | HTTP input and JSON/SSE framing |
| `routes` | Endpoints, registration, and queue admission |
| `request` | Options, session input, and context validation |
| `registry` | Request IDs and cancellation state |
| `worker` | Scheduling, state switching, and memory admission |
| `sessions` | History, sampling state, and transactional turns |
| `output` | Writer queues and final publication |
| `response` | Chat and text response formats |
| `tool_call` | Incremental tool-call parsing and argument normalization |
| `failure` | HTTP error classification |
| `stats` | Usage shared by active requests and committed sessions |

`src/prompt.rs` compiles the checkpoint template once. Cache boundaries must
match prefixes of the complete rendered prompt. Sessions pass typed settings
and messages to the worker.

## Ownership and scheduling

| Owner | State |
| --- | --- |
| GPU worker | Weights, expert pool, pipelines, scratch, and live sequence |
| Active request | Checkpoint, prefill position, pending token, drafts, sampler, decoder, and output count |
| Retained session | Committed messages, sampling settings, RNG, and context limit |
| Prefix cache | Reusable checkpoints and prompt logits |
| Response writer | Bounded queue, socket, and completion decision |

The worker runs requests round-robin, yielding after one prefill chunk or
complete decode step. Switching requests saves sequence state to CPU vectors
and restores the next request. A single active request stays on the GPU.
Events and expert residency are shared and are never restored from checkpoints.
Requests are interleaved, not batched together on the GPU.

Checkpoints contain KV and QSA state, DeltaNet state, convolution/PLE history,
tokens, position, and MTP state. Scratch is reusable after GPU completion.
Admission reserves checkpoint growth and sampler workspace; requests wait
when their combined reservations cannot fit.

## Sampling

Each request owns its RNG. Selection applies occurrence penalties, temperature,
top-k, then top-p. Greedy, unpenalized generation uses GPU argmax and adaptive
MTP. Sampling or penalties disable MTP verification.

A suspended request keeps its selected next token. It is not sampled again,
and other requests cannot consume its random draws. Output-length boundaries
do not draw unused tokens. Sessions continue their committed RNG unless given
a new seed; greedy turns consume no draws. Penalty counts come from the full
rendered prompt.

Exact-prefix hits return logits for the request's own sampler. Cache reuse
checks draft compatibility and MTP's following-token dependency. It never
reuses another request's sampled token or RNG.

## Cancellation and commit

Registration reserves an ID through final output. Cancellation can be set
without entering the GPU worker. The worker checks before admission and work,
and after prefill chunks. In-flight GPU work finishes before state is reused.

A turn pins committed history, settings, and RNG while generating provisional
output. Preparing completion reserves space for the new history. Immediately
before the final response, the writer atomically chooses completion or
cancellation. Only completion commits the prepared session state.

Cancellation or failure releases the reservation and leaves the previous turn
intact. A retry starts there. Disconnects after commit cannot undo the turn.
Write failures, full queues, and socket timeouts cancel unfinished requests.

## Retention

Sessions and prefix checkpoints have separate byte, count, and idle limits.
In-flight sessions are pinned; idle sessions may be evicted. History can
survive checkpoint eviction and be prefilled again. Neither store persists
across server restarts.

GPU context buffers are allocated once at the server maximum. A smaller
request limit reduces admission and checkpoint reservations, not GPU buffers.

## Limits and tests

Tool calling uses the checkpoint template and XML-style output parser. The
client executes the returned calls; sessions retain them for later turns.
The worker tracks raw decoded bytes for reconciliation and response limits,
and accumulates parsed content separately. Final tokenizer bytes pass through
the parser before it closes. Streaming text, JSON content, and committed
session content use the same accumulated text. Recovered calls preserve a
`length` finish reason when generation reaches its token budget.
See [tool calling](running.md#tool-calling) for supported request controls.

Effort controls, the Responses API, sampled MTP, and cross-request GPU batching
are unsupported. Chat uses the checkpoint template with thinking disabled.

Tests cover sampler continuation, pending tokens, commit/cancel ordering,
retention, memory admission, state restoration, and interleaved HTTP streams.
Worker tests also cover tool-call content across streaming, JSON, and sessions,
fallback delivery, final tokenizer bytes, and truncation finish reasons.
See [validation](validation.md) for commands. Concurrency throughput has not
been established by these correctness tests.
