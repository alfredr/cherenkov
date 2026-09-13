# Storage and downloads

`cherenkov paths` prints storage paths and the default model reference.
`downloaded_model` is the path used by the deprecated download command.

| Contents | Default path | Override |
| --- | --- | --- |
| Checkpoints and packed stores | `~/.local/share/cherenkov` | `XDG_DATA_HOME` |
| Transfer scratch | `~/.cache/cherenkov` | `XDG_CACHE_HOME` |
| Server configuration | `~/.config/cherenkov/cherenkov.toml` | `XDG_CONFIG_HOME` |

These defaults also apply on macOS. Each XDG override must be absolute;
Cherenkov appends `cherenkov` to it. Empty or relative values are ignored.
Resolving paths does not create directories. Downloading and packing create
the required parent directories.

`--root DIR` places data in `DIR`, scratch in `DIR/scratch`, and configuration
in `DIR/cherenkov.toml`. Put it after the subcommand:

```sh
cherenkov paths --root /Volumes/Models/cherenkov
```

An explicit model path overrides the managed default. `serve` reads the default
config if present, or the file selected by `--config`. Its `[server].root` can
change model lookup; a CLI root takes precedence. Other subcommands do not read
TOML. See [server configuration](server-config.md) for path resolution.

Use the [model index](model-index.md) to name checkpoints, prepare HF sources
without retaining their downloads, and collect unused managed stores.

## Prepared storage

```text
data/
  index.json
  artifacts/<id>/
    config.json
    tokenizer.json
    manifest.json
    dense.bin
    ngram.bin
    experts.bin
    experts2.bin + manifest2.json
    experts3.bin + manifest3.json
  leases/<id>
```

Prepared stores and retained source downloads live under `artifacts/`.
External checkpoints stay in their original locations. Clearing transfer scratch
leaves prepared stores intact. `model gc` removes unreferenced owned artifacts;
see the [index layout](model-index.md#storage-and-interfaces).

## Deprecated download command

The [getting-started workflow](../README.md#get-started) uses `prepare SOURCE`
to create an indexed store. The standalone `download` command is
deprecated but still works. Use `prepare --keep-source` to retain downloaded weights.

The default checkpoint is `Sawfwair/Qwen3.8-Flash-Next-MLX-4bit` at
`6cc9bbc0fae9ce26b7670b3ed1e26d557c154506`. Branches and tags passed to
`download --revision` are resolved to full commits. Other downloads print
their model path; pass it to `prepare` to register and prepare it.

`download --metadata-only` fetches config and tokenizer files without weights.
A later full download reuses them. The downloader validates the architecture,
reads shard names from the safetensors index, and checks available disk space.
This checkpoint needs about 104 GB for source weights and another 104 GB for
base packing.

Downloads use the Rust `hf-hub` client and Xet. Supply credentials through
`HF_TOKEN`, an existing Hugging Face login, or `download --hf-token`.
`HF_TOKEN` avoids exposing a token in command arguments. Tokens are not saved
in Cherenkov config or manifests.

Xet scratch uses `scratch/xet/`, unless `HF_XET_CACHE` is set. Credential
lookup follows `HF_TOKEN`, `HF_TOKEN_PATH`, and `HF_HOME`. Authentication grants
account access and rate limits; it does not guarantee faster transfers.

## Prepare

```sh
cherenkov prepare /path/to/model
cherenkov prepare /path/to/model --experts 2,3
```

Preparation publishes an indexed artifact under `data/artifacts/`.
`--output DIR` exports to a new directory and records it in the index. An existing
packed directory can also be registered as input. Omitting the source selects
the default HF checkpoint, also selected by bare `serve`. `pack` remains an alias
for `prepare`.

`--experts` accepts 4, 3, or 2, separated by commas, spaces, or repeated flags.
The default is 4. Low-bit targets require the Q4 base, built first if absent.
Missing variants share one pass through the base records. Valid stores are
reused; unselected stores are left intact. Allow about 39 GB for 2-bit and
54 GB for 3-bit in addition to the base store.

The packer checks disk space and publishes manifests after flushing output.
Inference also builds missing low-bit stores. `--repack` during inference
rebuilds the selected low-bit store.

The server's prefix cache and sessions are held in RAM. They have no disk store.

## BF16 import

`prepare` also accepts native Hugging Face BF16 checkpoints for `qwen4_exp`.
It splits fused expert matrices and quantizes them directly into the Q4 store,
without an intermediate checkpoint. Routers, convolutions, and norm vectors
remain BF16. The packer folds zero-centered norm offsets into their weights.
N-gram quantization groups follow the row width.

```sh
cherenkov prepare /path/to/bf16-model --experts 4,3,2
```

The packed directory keeps the tokenizer and configuration. If the checkpoint
has no MTP weights, its packed configuration disables drafting; the source
configuration stays unchanged. Existing MTP weights are retained.

## Checkpoint inspection

```sh
cherenkov inspect /path/to/model
cherenkov inspect /path/to/model/packed
cherenkov inspect /path/to/model.gguf --json
```

`inspect --json` reports tensor shapes, encodings, byte ranges, and preparation
requirements for the Qwen4-exp engine. Inspection uses the prepared artifact
when one is registered; otherwise it uses the source description.

The `cherenkov-model-data` workspace crate reads safetensors and GGUF without
Metal. Safetensors is a container; MLX affine quantization is a separate
encoding convention. The MLX adapter combines codes, scales, and biases into
logical tensor descriptions. GGUF block formats retain their own encodings.
The Qwen adapter assigns tensor roles and describes n-gram hashing and shards.

Sources expose object IDs, sizes, streaming byte reads, and optional mappings.
IDs are local to a source; callers must rebind them when copying to another store.
Mapped views keep their mappings alive after the source is dropped. Callers must
keep the files unchanged and, for indexed stores, hold the artifact lease until
all views are finished. The lease protects the store from garbage collection.
Strides describe expert records and interleaved n-gram rows without copying them.

The reader preserves configuration, shard metadata, and unknown encodings.
Inspection does not imply execution support: other architectures and GGUF
conversion remain unsupported by the Qwen4-exp engine. This layer does not
allocate GPU pools or manage model ownership and garbage collection.
