# Validation

Install the pinned tools once, then choose a check group. Run native tests,
coverage, and Metal checks without concurrent inference or benchmarks:

```sh
mise install
mise run hooks
mise run check
mise run check:full
```

| Group | Checks | Platform |
| --- | --- | --- |
| `mise run check:portable` | Rust formatting and spacing, Markdown, workflows, hook and task configuration | Linux or macOS |
| `mise run check` | Portable checks and all Rust targets with dead code denied | macOS |
| `mise run check:native` | Rust target checks followed by serial workspace tests | Apple Silicon |
| `mise run check:docs` | Documentation builder tests and the complete book/API build | macOS |
| `mise run check:full` | Portable and native checks, Metal syntax, and the complete site | Apple Silicon |
| `mise run coverage` | Instrumented tests, HTML and LCOV reports, and totals | Apple Silicon |

CI and pre-commit call the same tasks in `mise.toml`. The full group runs unit
tests once, then Metal checks and the site build; lightweight portable checks
can run in parallel. Coverage remains separate because it runs an instrumented
test suite. Run individual checks with `check:fmt`, `check:spacing`,
`check:markdown`, `check:workflows`, `check:hooks`, `check:tasks`, `check:rust`,
or `check:metal`.

Checks report failures without changing source files or installing tools.
Rerun `mise install` when tool pins change. Rust compilation commands use
`--locked` consistently; set `CARGO_NET_OFFLINE=true` for offline checks after
dependencies have been downloaded.

The additional `check:clippy` and `check:oxisym` diagnostics are separate from
the passing groups because they still report existing findings described below.
Oxisym selects its own nightly toolchain and requires two additional tools:

```sh
cargo install --locked cargo-dylint dylint-link
```

| Command | Purpose |
| --- | --- |
| `mise run fix` | Apply Rust formatting, spacing fixes, and Markdown fixes |
| `mise run fmt` | Format Rust |
| `mise run fix:spacing` | Format Rust and separate statement groups |
| `mise run fix:markdown` | Apply Markdown lint fixes |
| `mise run clangd` | Regenerate Metal editor configuration |

Review automatic spacing changes: the rules group syntax, not meaning.
Existing task names such as `fmt-check`, `lint-md`, `check-metal`,
`clippy`, `oxisym`, `fix-spacing`, and `fix-md` remain aliases.
The compatibility command `lint-spacing` runs `check:fmt` before `check:spacing`.

## Tests

Engine unit tests are child modules in `tests/unit/`, mirroring the source.
Container, byte-source, and discovery tests are in `crates/model-data/tests/`.
Automation tests are in `xtask/tests/`.

Pull requests run portable pre-commit checks on Ubuntu and
check all workspace targets and run the test suite on macOS 15 and 26.
The macOS jobs include synthetic Metal tests, then explicitly check Metal syntax
and the generated clangd configuration. Real-weight checks need a local model.

GitHub Actions caches Cargo downloads and compiled dependencies after
successful runs. Test caches are separate for macOS 15 and 26; docs, coverage,
and author checks use separate caches. Rust versions, Cargo manifests, and
lockfiles contribute to cache keys. Workspace crates are rebuilt, and the docs
build regenerates the API reference while reusing dependencies from
`target/site-rustdoc`. The first run for a new cache is cold; later compatible
runs can restore it.

Read the contribution terms in [AUTHORS](../AUTHORS), then acknowledge them
by adding your own entry as `Name <git-email> (@github-login)`. You can do this
in your first PR and use a GitHub noreply address. Later PRs reuse that entry.
The check matches the PR author's GitHub login; reviewers confirm that new
entries were added by the contributors themselves.

Run the same check locally with `cargo xtask check-author YOUR_GITHUB_LOGIN`.

| Area | Coverage |
| --- | --- |
| CPU | Config, CLI, packing, quantization, paths, and context limits |
| Model index | Registration, shared artifacts, leases, garbage collection, exports, and variant repair |
| Server | HTTP, sampling, RNG continuation, cancellation, eviction, and memory admission |
| Metal | Attention, argmax, experts, prefill, and complete state restoration |
| Automation | Timing, SVGs, cycle detection, resume, cleanup, and reports |

Metal tests require Apple Silicon. Real-weight tests use `CHERENKOV_MODEL_DIR`
or the managed model and skip when it is absent. Low-bit checks skip absent
Q2/Q3 stores. `check-metal` compiles the engine's assembled libraries and
rejects stale `kernels/.clangd`.

Prompt tests compare 20 independent template references. Text and error checks
need no model. Token checks need the matching tokenizer, but no weights or GPU.
See [fixture provenance](../tests/fixtures/prompt/README.md).

## Coverage

Run `mise run coverage` on Apple Silicon to run instrumented workspace tests
serially and generate reports in `target/coverage/`:

- `html/index.html`: browse coverage by file and source line.
- `lcov.info`: import coverage into an editor or another reporting tool.
- `summary.txt`: per-file and overall coverage totals.

Use `mise run coverage:report` to regenerate reports without rerunning tests.

Install the tools with `mise install rust cargo:cargo-llvm-cov`. If Rust is already
installed without coverage support, run `rustup component add llvm-tools-preview`.

The Rust coverage job runs on macOS 26 for pull requests and `main`. Each run
records its summary in GitHub Actions and retains the reports for 90 days in a
`rust-coverage-<commit>` artifact. Compare run summaries to track coverage changes.
There is no minimum coverage threshold yet.

Coverage measures Rust lines, regions, and functions. Test source files are
excluded from the totals. It does not measure Metal shader execution or doctests.
CI has no model weights, so model-dependent tests skip; their unexecuted Rust
paths still count toward coverage. Local runs with model weights may cover more.

## Pre-commit hooks

After `mise install`, run `mise run hooks` once to install the Git pre-commit hook.
It checks Rust formatting, statement spacing, Markdown, workflow syntax, and all
Rust targets with dead code denied. Checks run when matching files are staged;
they report failures without modifying files. Rust checks require macOS.

Run `mise run pre-commit` to check all tracked files before opening a PR. The
Ubuntu CI job runs the portable group; the macOS jobs run the native group
followed by `check:metal`.
Coverage and the test suite run separately from the commit hook.

Linked Git worktrees share the installed hook. Installation allows a missing
configuration so worktrees on branches without `.pre-commit-config.yaml` can
still commit normally.

## Live checks

```sh
cargo xtask smoke server --model /path/to/model
cargo xtask smoke control --model /path/to/model
cargo xtask smoke sessions --model /path/to/model
cargo xtask smoke download
```

Each server check starts and stops its own process. Checks cover JSON/SSE,
prefix reuse, reloads, concurrent sessions, cancellation, rollback, limits,
and recovery. The download check fetches metadata only and checks reuse.
Reports go to ignored `results/*-smoke.json` files.

## Lint limits

All workspace packages deny Rust's `dead_code` lint. `mise run check:rust` runs
`cargo check --locked --workspace --all-targets`, including libraries, binaries,
and tests. Unused private items fail the check. Public library APIs may be used
by downstream crates, so this does not detect every unused public API.

Clippy uses cognitive complexity 12 and nesting 2, and checks unnecessary
`else` branches. Use guard clauses and helpers without obscuring numerical
or dispatch order.

Complexity passes. The experimental nesting limit still flags existing code.
To run the other lints independently:

```sh
cargo clippy --workspace --all-targets --offline -- -D warnings -A clippy::excessive_nesting
```

Oxisym reports existing structural-similarity findings for manual review.

## Known gaps

Use the [benchmark suite](../benchmarks/README.md) for performance checks.
Saved results apply to their recorded revisions.

- A local AC comparison found a possible 6.5% long-prompt decode regression
  after readability changes. It remains unresolved.
- Full download throughput and a fresh full-size pack have not been validated.
- Cached and fresh runs can differ on close argmax decisions as expert
  accumulation order changes. Exact state restoration does not prevent this.

## BF16 fixture

The optional [Git LFS fixture](../tests/fixtures/qwen4_exp/README.md) exercises
BF16 packing, Metal prefill and decode against the CPU reference, and CLI
generation. It also checks mapped views, indexed source removal, variant
replacement while readers hold leases, and external exports after collection.
The suite also covers `store add`, `prepare`, and generation by alias.
It includes DeltaNet, sparse attention, routed experts, and n-grams;
it has no MTP weights. Full-size BF16 import has not been run locally.

## Loading checks

The model-data tests run without Metal:

```sh
cargo test -p cherenkov-model-data
cargo test -p cherenkov-model-data --test remote -- --ignored --test-threads=3
```

The remote tests read every safetensors header at pinned Hugging Face
revisions and check it against the shard index. They fetch byte ranges only,
with no weight downloads. All six cases passed locally:

| Checkpoint | Stored tensors | Encoding coverage |
| --- | --- | --- |
| Sawfwair Qwen3.8-Flash-Next MLX | 3,817 | Affine Q4, groups 64 and 32 |
| Qwen3.8-Flash-Next | 1,658 | BF16 |
| Qwen3.8-Flash-Next 0.2B MoE | 271 | BF16 |
| mlx-community Qwen3.8-27B | 2,180 | Affine Q4, group 64 |
| DeepSeek-V4.1-Flash | 96,085 | Mixed storage; FP8 encodings preserved |
| s-zaizen DeepSeek-V4.1-Flash NVFP4 | 188,245 | Mixed storage; FP8 encodings preserved |

These checks validate metadata and byte ranges, not weight contents or
inference. DeepSeek's quantization config and stored components are retained;
its FP4/FP8 decoding is not implemented. The local MoE fixture additionally
tests mapped weights, native-to-packed conversion, and CPU/Metal agreement.
The test definitions contain the repository names and full revisions.
