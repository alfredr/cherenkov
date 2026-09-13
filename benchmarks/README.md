# Benchmarks

The runner accepts an indexed alias, source URI, or prepared directory. It uses
the index to find the artifact and accepts both flat stores and older
`model/packed` layouts. Use `--root DIR` if the model is in a separate index.
Model inspection runs outside the timed inference capture.

```sh
cargo xtask bench flash --build-stores --update-readme
cargo xtask bench disk://models/owner/repo --root /path/to/index
```

The suite requires Rust, Metal, and a packed checkpoint. Run on AC power with other
inference, builds, and GPU tests stopped. Prefix commands with
`mise exec --` if Mise is not active in your shell.

```sh
cargo xtask bench /path/to/model --dry-run
cargo xtask bench /path/to/model --build-stores
cargo xtask bench /path/to/model --cases code,prose --rounds 1
cargo xtask bench /path/to/model --output results/comparison
cargo xtask bench /path/to/model --output results/comparison --resume
cargo xtask bench /path/to/model --mode light --note "M5 Pro 48 GB, idle"
cargo xtask bench /path/to/model --mode heavy --build-stores --note "M5 Pro 48 GB, idle"
```

The runner builds offline unless given `--binary`. `--build-stores` permits
missing low-bit stores to be built during load. Otherwise missing stores are
an error. Allow an extra 39 GB for Q2 and 54 GB for Q3. Loading and conversion
are excluded from prefill and decode timings.

## Sharing a run

Two presets exist for sending a run to someone else. `--mode light` runs one
round of the code, prose, and long-prefill cases on whichever expert stores
are already built, about ten samples, and takes a few minutes. `--mode heavy`
runs the whole suite including pelicans, which takes hours and needs every
store or `--build-stores`. Both zip the finished results directory beside
itself (`--archive` does the same for any run), so one file holds
`report.json`, `summary.md`, the gallery, every answer, and the pelicans.
Attach it to a pull request or issue. Explicit `--configs`, `--cases`, and
`--rounds` override a preset's choices.

Light mode consults the model index, including for `--dry-run`.

Each report records the machine in `provenance.hardware_detail`: kernel,
memory, CPU thread count, and the capacity and free space of the volume
holding the model, all read through libc; on macOS also the chip, GPU core
count, OS version, and NVMe model and capacity from `system_profiler`; and a
two-gigabyte uncached read sample of the expert store in GB/s taken before
the first sample. Automatic machine metadata excludes serial numbers,
device identifiers, the hostname, and home-directory paths. The runner's
binary and model arguments appear as `<binary>` and `<model>`; the runner's
`--root` path appears as `<root>`. Custom suite
prompts and configuration arguments, generated answers, and `--note` text
are preserved verbatim. `--note` records conditions such as power or other load.
`settings` records the context capacity and cap overrides; the suite path
appears as `<suite>` and its content hash is recorded in the signature.
The server TOML is never read, and the pool is adaptive unless a
configuration passes `--pool-gb`. The summary's first line repeats the
hardware facts so runs from different machines can sit side by side.

## Suite

[suite.json](suite.json) defines 80 fresh-process samples:

- Six completion workloads × four settings × three rounds.
- One long-prefill workload and one pelican per setting.

Settings are Q4, Q4 residents with Q2 misses and cut 0.08, Q3, and Q2. All use
two adaptive drafts, an adaptive expert pool, and 8,192-token context. The mixed
setting's cut makes output timing-dependent. Configurations rotate between
rounds; pelicans run last.

Answers run to EOS. Safety caps are 4,096 tokens, except LRU, reasoning, and
pelicans at 7,168, and document summaries at 1,024. Capped or cycling answers
are saved but excluded from medians. Completion does not establish correctness.
Compare answer length and completion time alongside tg/s.

Expert precision applies to prefill and decode. The saved September 9 baseline
used Q4 batched prefill in every mode; later prefill comparisons are separate.

## Measurement

Each sample uses a fresh process without `--check`, warm repeats, prefix
caching, or developer overrides. Power is checked
at process boundaries and every 30 seconds. A power-source change invalidates the
sample; shorter transitions may be missed.

The engine sizes its expert pool from Metal's recommended working set after
fixed buffers and reservations, and fits prefill chunks to available memory.
The suite keeps prompts, answer caps, and context capacities fixed for
comparison across machines. Custom suites can change those workloads;
`--case-cap` changes answer caps and `--mode light` reduces the sample count.

Reported GPU memory above the target machine's physical memory invalidates
a sample. Memory is read after prefill scratch is released.
`gpu_span_ms` is the interval between the
command buffer's GPU start and end timestamps, including gaps. `io_wait_ms`
measures host servicing of expert reads. The intervals overlap and must not
be added together. Neither measures GPU utilization.

## Reports and resume

| File | Contents |
| --- | --- |
| `report.json` | Samples, arguments, phase timings, power readings, and source/model/binary hashes |
| `README.md` | Optional run observations, preserved during regeneration |
| `summary.md` | Medians, pelicans, and links to full answers; renders on GitHub and the site |
| `outputs/` | Full generated answers |
| `pelicans/` | Unedited, XML-validated drawings |
| `gallery.html` | Local browser preview, regenerated from the report |

Browse the [saved reports](../results/README.md) on GitHub or the documentation
site. Published runs retain the report, summary, observations, outputs, and
pelicans. The HTML gallery is a local preview.

Results default to `results/<UTC timestamp>/` and are written after each sample.
New runs are ignored by Git. Use `git add -f results/<name>` to retain one.
SVG rates are reported separately; malformed or incomplete SVGs are invalid.

`--resume` requires matching binaries, indexed model identity, metadata, suite,
and selections. Reports store the model ID rather than its local path. Older
reports can migrate when their path or path hash matches the indexed local
source. Reports containing only `<model>` without an identity require a new run.
It skips successful and content-invalid samples. Engine errors, memory-limit failures,
and power changes stop the suite and are retried on resume.

To retry capped answers, raise their caps with, for example,
`--case-cap code-lru=7168 --resume`. Completed EOS samples remain. Earlier
attempts and signatures are retained. Lower caps or other setup changes are
rejected.

Every 30 seconds, cycle detection looks for four exact repetitions of a
32–512-word block at the output tail. A match saves the evidence, ends that
sample, and continues the suite. The check does not detect every kind of loop.

Regenerate reports without inference:

```sh
cargo xtask summarize results/<name>
cargo xtask readme results/<name>
```

`summarize` preserves the run's README. `--update-readme` runs the second
command after a completed benchmark.

## Paired prefill

```sh
cargo xtask prefill --before /path/to/old/cherenkov \
  --after target/release/cherenkov --model /path/to/model \
  --out results/prefill-comparison --rounds 2
```

This compares three prompt lengths with Q4/Q3/Q2 experts and alternating
binary order. `--configs 4/2` or `--configs 4/3` selects mixed precision without
a deadline cut. Each sample generates eight tokens to check the decode handoff;
it does not measure complete answers or decode throughput.

Prepare the selected low-bit stores first. Any store build invalidates a
sample. Reports retain hashes, source diff, prompts, telemetry, output, and
power readings.
