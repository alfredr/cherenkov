# Cherenkov

Cherenkov is a Rust and Metal inference engine for experimental Qwen4 MoE models
(`qwen4_exp`) on memory-constrained Apple devices. It runs from the 4-bit
quantized checkpoint, streaming experts from SSD into a bounded GPU cache.
The saved benchmark ran on a 32 GB M4 MacBook Air and used about 21 GB
of Metal allocations.

## Get started

Cherenkov runs on Apple Silicon and macOS. To build from source, install the Xcode
command-line tools and [Mise](https://mise.jdx.dev/).

```sh
mise install
mise exec -- cargo build --release
target/release/cherenkov prepare hf://Sawfwair/Qwen3.8-Flash-Next-MLX-4bit@6cc9bbc0 --name flash
target/release/cherenkov serve --model flash
```

Cherenkov accepts native BF16 checkpoints or MLX affine 4-bit weights
quantized in groups of 64 (32 for the 160-column n-gram table). We tested
[Sawfwair/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit/tree/6cc9bbc0fae9ce26b7670b3ed1e26d557c154506)
at revision `6cc9bbc0`, used in the commands above.

> [!IMPORTANT]
> The
> [mlx-community conversion](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit/blob/main/config.json)
> uses groups of 32 for the main weights, so Cherenkov cannot load it.

The built-in packer converts BF16 weights to 4-bit as it writes aligned records.
It preserves existing MLX quantized weights bit for bit. No Python or MLX runtime
is required. `prepare` registers the source, downloads it, and prepares the model.
Allow roughly 210 GB during preparation. Packing removes the temporary source
after success; add `--keep-source` to retain it.
See [storage and downloads](docs/storage.md) for paths and `HF_TOKEN`.

For a checkpoint already on disk, register its parent store once:

```sh
target/release/cherenkov store add models /path/to/models
target/release/cherenkov prepare disk://models/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit --name flash
target/release/cherenkov serve --model flash
```

This example uses `/path/to/models/Sawfwair/Qwen3.8-Flash-Next-MLX-4bit`.
A direct checkpoint path also works. The alias is optional; commands accept the
source URI too. See the [model index](docs/model-index.md) for HF cache stores.

## Benchmarks

<!-- benchmarks:start -->

The benchmark ran on Apple M4 hardware with 32 GiB of memory.
The engine reported 20.98 GB of Metal allocations.
The report contains 80 valid samples from revision `93c514f`.

The inference rates exclude loading and store construction. Answer lengths vary,
so compare completion times in the full report.

[Full report](results/baseline-2026-09-09/report.json).

| Experts | code tg/s | code-lru tg/s | debug-bisect tg/s | prose tg/s | reasoning tg/s | structured tg/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-bit | 8.52 | 7.07 | 7.06 | 7.71 | 6.77 | 7.88 |
| 4-bit / 2-bit misses + cut | 8.29 | 8.03 | 7.96 | 8.92 | 8.05 | 9.33 |
| 3-bit | 12.13 | 10.72 | 10.27 | 9.83 | 10.86 | 12.67 |
| 2-bit | 17.38 | 15.23 | 13.54 | 11.73 | 13.00 | 20.17 |

| Experts | prefill-long pp/s |
| --- | ---: |
| 4-bit | 83.20 |
| 4-bit / 2-bit misses + cut | 77.80 |
| 3-bit | 75.80 |
| 2-bit | 78.90 |

Settings with a deadline cut are not reproducible.

[All timings and outputs](results/baseline-2026-09-09/summary.md).

### Pelicans

These are unedited model outputs from the benchmark.

| 4-bit | 4-bit / 2-bit misses + cut |
| --- | --- |
| ![Pelican](results/baseline-2026-09-09/pelicans/exact-4bit.svg) | ![Pelican](results/baseline-2026-09-09/pelicans/misses-2bit.svg) |

| 3-bit | 2-bit |
| --- | --- |
| ![Pelican](results/baseline-2026-09-09/pelicans/all-3bit.svg) | ![Pelican](results/baseline-2026-09-09/pelicans/all-2bit.svg) |
<!-- benchmarks:end -->

Run the full suite, save its answers and pelicans, and refresh this section:

```sh
cargo xtask bench flash --build-stores --update-readme
# Regenerate from a completed run without inference:
cargo xtask readme results/baseline-2026-09-09
```

All outputs live in `results/`. See the [benchmark method](benchmarks/README.md),
[paired prefill measurements](results/prefill-lowbit-2026-09-09/README.md),
and [current validation](docs/validation.md).

## Run

**Server:** `serve` keeps the model loaded and caches repeated prompt prefixes.
Connect an OpenAI-compatible client to `http://127.0.0.1:8080/v1` with model
`cherenkov`. Chat and text completions support streaming and sampling.
The server interleaves up to two active requests by default, with cancellation
and optional retained sessions. Chat renders the checkpoint's Jinja template
with thinking disabled. Reasoning-effort controls are not exposed yet.
See the [HTTP API](docs/running.md).

**CLI:** pass an indexed model or local directory and a prompt to generate directly.

```sh
target/release/cherenkov flash \
  'Explain hash collisions.' --max-tokens 256
target/release/cherenkov flash \
  'Explain hash collisions.' --experts 3
target/release/cherenkov status
target/release/cherenkov --help
```

List registered models and inspect their stores:

```sh
target/release/cherenkov model list
target/release/cherenkov inspect flash
```

`model remove <reference>` releases the registration. `model gc --dry-run`
previews unused managed stores; `model gc` deletes them. See the
[model index](docs/model-index.md) for local sources, exports, and retention.

### Statistics

Query the running server from another terminal:

```sh
target/release/cherenkov dash
target/release/cherenkov stats summary
target/release/cherenkov stats layers
target/release/cherenkov stats experts 7
target/release/cherenkov stats layers --json
```

`stats` shows formatted summaries; `--json` returns all fields. In `dash`,
select a row for details. Press Enter on a layer to inspect its experts,
then Escape to return. See [statistics](docs/stats.md) for paging and controls.

## Options

The default is **4-bit experts with two adaptive speculative drafts**.

| Option | Effect |
| --- | --- |
| `--experts 4\|3\|2` | Routed-expert precision in prefill and decode. Lower precision trades accuracy for speed. |
| `--miss-experts 2` | Fetch new misses at 2-bit in Q4 mode. |
| `--drafts N` | Speculative drafts, 0–3; `0` disables speculation. |
| `--temperature T` | Sampling temperature; default `0` is greedy. Sampling disables MTP verification. |
| `--top-k K`, `--top-p P`, `--seed N` | Filter and seed sampling; defaults are unfiltered and unseeded. |
| `--max-tokens N` | Maximum generated tokens; default 64. |
| `--max-ctx N` | Context capacity; default 2,048. Larger contexts leave less memory for experts. |
| `--pool-gb N` | Expert-pool memory budget; adaptive by default. |
| `--raw` | CLI only: use the prompt without the chat template. |
| `--cut-weak W` | Skip late weak experts; output then depends on disk timing. Off by default. |

Prepare low-bit expert stores ahead of inference:

```sh
target/release/cherenkov prepare flash --experts 3
target/release/cherenkov prepare flash --experts 2
target/release/cherenkov prepare flash --experts 2,3
```

Replace the indexed reference with a local checkpoint path to pack it directly.
The 4-bit base is built if needed and retained; selected low-bit stores coexist
beside it. Allow about 39 GB extra for 2-bit, 54 GB for 3-bit, or 93 GB for both.
Existing stores are reused. Inference also builds a missing variant on
first use.

[Server configuration](docs/server-config.md) covers TOML defaults, memory
limits, cache policy and reloads. See [running options](docs/running.md)
for the full interface.

## How it works

These are active parts of the default engine:

- **MTP speculation.** The checkpoint's own multi-token prediction head proposes
  up to two tokens. The trunk verifies them together and commits the accepted
  prefix, restoring recurrent state after a rejection. The first draft shares
  the trunk's command buffer; a second is chained after full acceptance.
- **Speculative routing.** A one-block lookahead predicts which experts the next
  block will need and starts background reads. Actual routing still determines
  which experts run. Required misses take priority over speculative reads.
- **Expert cache.** An adaptive, resident LRU pool keeps recently used experts
  in unified memory. The GPU computes cached experts while CPU threads read
  missing records directly into free pool slots.
- **Block address tables.** Each block forms the union of experts needed by its
  token and draft rows. A table maps those experts to GPU cache addresses and
  tags their precision; separate per-row weights preserve each token's routing.
  Kernels follow the table, so experts can change cache slots without moving
  the rest of the model.
- **Shared page mappings.** Packed dense weights are memory-mapped and exposed
  to Metal without a second copy. The expert pool also shares CPU/GPU pages:
  disk reads fill the same memory the kernels consume. Events keep the GPU
  from reading unfinished records and the CPU from overwriting active slots.
- **Custom Metal kernels.** Quantized projections, expert dispatch, sparse
  attention, DeltaNet, PLE and MTP run in native kernels. Longer prompts use
  batched matrix kernels and a bounded expert streaming ring.
- **N-gram offloading.** The large n-gram embedding table stays in an SSD-backed
  mapping. CPU threads prefetch the selected rows through the OS page cache,
  then dequantize them into small shared buffers for the GPU's PLE blocks.
  Only those gathered embeddings occupy GPU buffers.
- **Prefix caching in server mode.** Repeated prompts restore attention,
  recurrent and MTP state, then process only the uncached suffix. Memory,
  entry count and idle expiry are bounded.

**Lower-bit experts are opt-in.** `--experts 3` or `--experts 2` compresses
routed experts in both prefill and decode. With `--miss-experts 2`, Q4
lookahead reads continue normally, while unpredicted misses fetch smaller
2-bit records just in time. A fetched record keeps its precision while cached;
the block table selects the matching kernel. Shared experts and dense
projections retain their original precision. Lower-bit stores are derived
once from Q4 and reused; no quantization happens in the decode loop.

See the [engine guide](docs/engine.md) for the address-table layout and
synchronization. Direct file-backed expert residency is a separate developer
option; the default uses the shared pool described above.
The [documentation index](docs/README.md) maps the remaining guides and source.

## Development

Install the tools with `mise install`, then run:

```sh
mise run hooks       # install pre-commit checks
mise run check       # portable lints and Rust target/dead-code checks
mise run check:full  # also run tests, Metal validation, and the site build
mise run coverage    # instrumented tests and HTML/LCOV coverage reports
mise run fix         # apply Rust, spacing, and Markdown fixes
```

PRs run the same check groups on Ubuntu and macOS and save coverage reports.
See [validation](docs/validation.md) for the groups, platform requirements,
individual checks, and additional Clippy and Oxisym diagnostics.

## License

[MIT](LICENSE), with [third-party notices](docs/third-party-notices.txt).
Model weights have their own license.
