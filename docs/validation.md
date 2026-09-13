# Validation

Run checks without concurrent inference or benchmarks:

```sh
mise install
mise run lint-spacing
mise run lint-md
CHERENKOV_MODEL_DIR=/path/to/packed/model mise run test
mise run check-metal
mise run clippy
mise run oxisym
```

`mise.toml` pins the tools. Oxisym selects its own nightly toolchain and
requires two additional tools:

```sh
cargo install --locked cargo-dylint dylint-link
```

| Command | Purpose |
| --- | --- |
| `mise run fmt` | Format Rust |
| `mise run fix-spacing` | Format Rust and separate statement groups |
| `mise run fix-md` | Apply Markdown lint fixes |
| `mise run clangd` | Regenerate Metal editor configuration |

Review automatic spacing changes: the rules group syntax, not meaning.

## Tests

Engine unit tests are child modules in `tests/unit/`, mirroring the source.
Automation tests are in `xtask/tests/`.

Pull requests check Rust formatting and statement spacing on Ubuntu and
run the test suite on macOS 15 and 26. The macOS jobs include synthetic Metal
tests; real-weight checks need a local model.

GitHub Actions caches Cargo downloads and compiled dependencies after successful
runs. Test caches are separate for macOS 15 and 26; docs and author checks use
separate caches. Rust versions, Cargo manifests, and lockfiles contribute to
cache keys. Workspace crates are rebuilt, and the docs build regenerates the
API reference while reusing dependencies from `target/site-rustdoc`. The first
run for a new cache is cold; later compatible runs can restore it.

Read the contribution terms in [AUTHORS](../AUTHORS), then acknowledge them
by adding your own entry as `Name <git-email> (@github-login)`. You can do this
in your first PR and use a GitHub noreply address. Later PRs reuse that entry.
The check matches the PR author's GitHub login; reviewers confirm that new
entries were added by the contributors themselves.

Run the same check locally with `cargo xtask check-author YOUR_GITHUB_LOGIN`.

| Area | Coverage |
| --- | --- |
| CPU | Config, CLI, packing, quantization, paths, and context limits |
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
