# Engine

The Metal engine streams routed experts through a bounded cache. Dense weights
remain resident, and the CPU gathers n-gram rows from a file mapping.
The CPU path is the reference used by `--check`.
Neither path requires MLX at runtime.

## Source map

| Code | Purpose |
| --- | --- |
| `src/qwen4_exp/config.rs`, `manifest.rs`, `packed.rs` | Model and store layout |
| `src/qwen4_exp/pack.rs`, `lowbit.rs` | Base packing and low-bit conversion |
| `src/qwen4_exp/cpu.rs` | Reference equations |
| `src/qwen4_exp/gpu.rs` | Shared GPU types and state |
| `src/qwen4_exp/gpu/load.rs`, `params.rs` | Allocation, pipelines, and shader parameters |
| `src/qwen4_exp/gpu/decode.rs`, `mtp.rs` | Trunk and draft execution |
| `src/qwen4_exp/gpu/state.rs` | Rollback and checkpoints |
| `src/qwen4_exp/gpu/streaming.rs`, `residency.rs` | Routing, expert slots, and IO |
| `src/qwen4_exp/gpu/prefill/` | Batched prefill |
| `src/runner/decode.rs`, `src/sampling.rs` | Acceptance, sampling, and EOS |
| `src/runner/diagnostics.rs` | CPU comparisons and dumps |
| `src/metal.rs`, `src/kernels.rs` | Metal buffers and source assembly |

Decode and prefill have child modules for attention, DeltaNet, experts,
hyper-connections, PLE, and sampling. Tests mirror the GPU modules under
`tests/unit/qwen4_exp/gpu/`. See the [kernel map](../kernels/README.md) for shaders.

Host `repr(C)` structs must match their Metal definitions in field order and
type. Buffer-binding offsets are bytes; kernel tensor indices are elements
unless stated otherwise.

## Decode and expert IO

A step processes one token and up to three drafts in row-major buffers.
The residual has four hidden-width streams. MoE output may be deferred until
the next normalization; PLE applies any pending output before its transform.

Each block uses this handshake. Values are relative to its sequence number.

| Signal | Writer | Meaning |
| --- | --- | --- |
| `seq` | GPU | Router indices and weights are ready |
| `seq + 1` | CPU | Address table and row weights are ready; resident experts may run |
| `seq + 2` on `event_res` | GPU | Resident computation is complete; deadline boundary |
| `seq + 3` | CPU | Required reads have completed; fetched experts may run |

### Address tables

The CPU forms the union of experts selected by all token rows and resolves
each `(layer, expert)` pair to a cache slot. `slot_tab` holds GPU addresses;
`wmap` holds each row's routing weights, with zero for unused experts.
Entries are ordered by residency, then by maximum routing weight across rows.
The final two table entries hold resident and total expert counts.

Address bit 63 marks Q3 and bit 62 marks Q2. Kernels mask these tags before
loading a record and use them to select dequantization.

For a token routed to A/B and a draft routed to B/C, with B missing:

| Entry | Token weight | Draft weight | Ready |
| --- | --- | --- | --- |
| A | A's weight | 0 | Resident |
| C | 0 | C's weight | Resident |
| B | B's weight | B's weight | After read |

The GPU computes resident contributions while the CPU fills missing slots.
Current-step slots cannot be evicted. With `cut_weak`, late weak experts may
be removed from the active table, but their slots remain reserved until IO
completes. Output then depends on read timing.

Lookahead predicts the next block's routes. Its reads start after the current
block's required misses finish. Residency changes accumulation order and can
change rounding, so use fresh processes for output comparisons.

## Prefill

Long prompts run layer by layer with matrix projections. Tokens are grouped by
expert, gathered, projected, and scattered back. Frequently used experts stay
in the decode pool; others stream through a 64-record ring in groups of eight.

The GPU signals after consuming a group. The CPU waits before reusing its
slots, fills the next group, then signals readiness. Each prefill event has
one writer and increasing values.

Prefill and decode share record layouts and stores. Prefill uses Q4 or Q2/Q3
GEMMs with 8/16/32-token tiles, half weights, and float accumulation. Decode
uses specialized half-dot reductions.

Uniform low-bit mode fills both pool and ring from that store. Mixed mode
keeps resident precision, adds kept records at Q4, and reads transient misses
at `--miss-experts` precision. Ring slots retain Q4-sized spacing. Shared
experts and dense projections remain Q4. Deadline cuts apply only to decode.
The ring-wait metric measures CPU waiting for GPU consumption, not disk IO.

## State and rollback

The CPU stores each DeltaNet head as `S[key_lane][value_lane]`. After
normalizing q/k and applying the causal convolution, each token computes:

1. `S = decay * S`
2. `delta = beta * (v - S^T k)`
3. `S = S + k delta^T`
4. `y = S^T q`, then gated RMS normalization

The GPU stores the transpose. One simdgroup owns a value lane and distributes
128 key lanes across 32 threads, keeping four floats per thread in registers.
Reduction order and half projections can differ from the CPU reference.

Verification saves recurrent state and convolution history after candidate
rows. After partial acceptance, `commit(n)` restores row `n - 1` and advances
the logical position. Attention and PLE entries beyond it are ignored or
overwritten. MTP tracks its own KV length and following-token dependency;
prefix reuse checks both.

Preserve reduction order, accumulator precision, and unrolling when moving
code. These affect numerics and register pressure. MTP's expanded residual
uses an eight-row projection path even though verification has at most four rows.

## Memory ownership

The packed model owns dense mappings borrowed by `Gpu`. Expert pools own their
slot memory. Shared CPU/GPU pages require event or command-buffer synchronization
before reads or overwrites.

The default expert pool is a shared Metal allocation. CPU file reads fill the
same slots the GPU consumes. Dense weights use `newBufferWithBytesNoCopy` over
a file mapping. `CHERENKOV_POOL=set` instead wraps mapped expert regions and
supports Q4 only.

The n-gram store remains CPU-mapped. Token history determines rows to prefetch
at each decode step or prefill chunk. `Packed::ngram_row` dequantizes them into
small shared buffers; the GPU runs PLE projections, gating, and convolution.
The full table need not remain GPU-resident.
