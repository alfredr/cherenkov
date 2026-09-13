# Documentation

Run `mise run docs` on macOS to build the
[website](https://alfredr.github.io/cherenkov/) in `_site/`.
`.gitattributes` selects the files and `SUMMARY.md` sets the navigation.
Changes to `main` publish automatically.

- [Rust API](rust-api.md): generated library reference, including private items.
- [Running](running.md): CLI, HTTP API, prefix caching and precision modes.
- [Server configuration](server-config.md): TOML, memory policy and control CLI.
- [Statistics](stats.md): summaries, detailed JSON, and the live dashboard.
- [Storage](storage.md): paths, downloads and packed expert stores.
- [Engine](engine.md): model execution, expert streaming and synchronization.
- [Developer options](developer-options.md): diagnostic environment variables.
- [Validation](validation.md): tests, linters and known limitations.
- [Serving architecture](serving-design.md): ownership, scheduling and cancellation.
- [Benchmarks](../benchmarks/README.md): suite, reports and pelicans.
- [Saved results](../results/README.md): measurements and generated answers.

## Repository layout

| Directory | Contents |
| --- | --- |
| `src/` | CLI, server, storage and shared runtime support |
| `src/server/` | HTTP framing, routes, request policy, scheduling, sessions and output |
| `src/control/` | Control state and statistics protocol |
| `src/cli_output/` | Terminal reports and dashboard |
| `src/runner/` | Resumable decoding and optional diagnostics |
| `src/qwen4_exp/` | Model config, packer, CPU reference and GPU execution |
| `kernels/` | Metal fragments grouped by common primitives and model subsystem |
| `tests/unit/` | Engine child-module tests, mirroring the source hierarchy |
| `xtask/src/`, `xtask/tests/` | Rust automation and its tests |
| `xtask/templates/` | HTML template used to generate benchmark galleries |
| `benchmarks/` | Workload definitions and measurement instructions |
| `results/` | Reviewed measurements; new local runs are ignored |

Model weights and packed stores use the configured [data directory](storage.md).

## Building the documentation

Install Rust and mdBook with `mise install rust github:rust-lang/mdBook`, then run
`mise run docs` (or `cargo xtask docs`). The build generates documentation for all
workspace libraries, fails on rustdoc warnings, and publishes the complete
reference under `_site/api/` after building the book. It needs macOS for the
Apple framework dependencies; model weights are not required.

To preview the combined site, run `python3 -m http.server --directory _site 8000`
and visit <http://localhost:8000/>. Rebuild to pick up source changes.

For just the Rust reference, run:

```sh
cargo doc --locked --workspace --no-deps --document-private-items --open
```
