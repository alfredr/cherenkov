# Model index

The index records model sources and prepared stores. A source can be a local
checkpoint, an HF cache snapshot, a packed directory, or a Hugging Face repository.
Registration inspects the model; it does not imply execution support.

## Select a model

```sh
cherenkov prepare hf://owner/repo@revision --name small-moe
cherenkov serve --model small-moe
cherenkov model list
cherenkov inspect small-moe --json
```

`prepare` resolves the source through the index, downloads weights if needed,
and prepares it. `model add SOURCE` registers metadata without preparing weights.
An alias is optional. All model commands use the same selectors:

| Selector | Selects |
| --- | --- |
| `hf://owner/repo[@revision]` | HF repository, with an optional revision |
| `disk://store/owner/repo[@revision]` | Model in a registered filesystem store |
| `alias` or `id` | Existing index entry |
| `/path/to/model` or `./model` | Local checkpoint or prepared directory |

The owner remains part of the identity. An unqualified HF URI reuses its indexed
commit without contacting HF. If several commits match, specify `@commit`, an
alias, or an ID. Commit prefixes need at least eight hexadecimal characters and
must select one entry. Cached commit prefixes follow the same rule. Branches and
tags are resolved when explicitly requested; use the returned reference or alias
for later index operations.
Bare names only look up aliases and IDs; they never trigger a download.

URLs use literal repository names; queries, fragments, escaped names, and dot
segments are rejected. A revision may contain slashes, such as `@feature/branch`.
Quote filesystem paths containing spaces. Legacy `model:` selectors and the
`pack` command remain accepted for compatibility.

### Filesystem stores

```sh
cherenkov store add models /Volumes/Models
cherenkov store add cache ~/.cache/huggingface/hub --layout hf-cache
cherenkov prepare disk://models/owner/repo --name local-model
cherenkov prepare disk://cache/owner/repo@commit
cherenkov store list --json
cherenkov store disable models
cherenkov store enable models
cherenkov store remove models
```

The default layout is `ROOT/owner/repo`. `hf-cache` reads HF's
`models--owner--repo/snapshots/commit` layout and cached refs without downloading.
Registration records the root without crawling it. Disabling a store blocks its
`disk://` selectors; existing aliases and prepared models remain usable. Removing
a store deletes only its registration. Re-registering its name creates a new
store ID. Local files always remain externally owned.

## Register and inspect

HF registration pins the requested revision (default `main`) to a full commit.
It reads config, the shard index, and each safetensors header using byte ranges.
Small n-gram metadata arrays may also be read, but full shards remain remote. Servers
that ignore range requests are rejected. Remote GGUF registration is not supported.

Credentials come from `--hf-token`, `HF_TOKEN`, or the HF token file. The index
stores the repository, endpoint, and commit, but no credentials.

`model list` and inspection of existing entries work offline. Both accept
`--json`, including a
resolvable `reference` field. Text output uses the terminal's report renderer.
`inspect --json` includes tensor shapes, encodings,
and byte ranges. Inspection supports more architectures than the inference engine.
For an indexed model, `inspect` describes its prepared store if one is recorded;
otherwise it returns the source description saved at registration.

## Prepare and run

```sh
cherenkov prepare small-moe --experts 4,3,2
cherenkov small-moe "Explain this model."
cherenkov serve --model small-moe
```

Indexed packing writes a managed store. When an HF source is needed, packing
reuses its retained copy or downloads it to a temporary owned directory.
New downloads are removed after successful packing unless `--keep-source` is set.
Conversion needs room for the complete source and prepared output together;
it does not release shards as it proceeds.
`prepare --hf-token` supplies credentials when the registered source requires them.

Local sources and existing HF cache snapshots remain externally owned. Cherenkov
never deletes them. `prepare --output DIR` exports to a new, externally retained
directory outside the managed `artifacts/` directory and records it in the index.
Direct paths also resolve through the index. The deprecated `download` command
keeps its older layout; `prepare` can register and use its output.

Generation and serving require a prepared model. Missing expert variants are
built according to the existing packing policy. Preparation also fills missing
auxiliary metadata from an available local or retained source. Older
`model/packed` layouts can keep config and tokenizer in the model directory.
Each update publishes
a new store, so running readers keep using their original files. Managed stores
reuse unchanged binary files through hard links where possible. Variant availability
checks load the manifest and check for the Q4 base file. Low-bit checks also
validate layouts, sizes, and sample records. Automatic repairs follow the same
build policy as missing variants and preserve files used by existing readers.

A server can select an indexed model in TOML:

```toml
[server]
model = "small-moe"
```

`server.model` and `server.model_dir` are mutually exclusive. CLI model selection
replaces the TOML selection. Changing either setting requires a server restart.

## Remove and collect

```sh
cherenkov model remove small-moe --source-only
cherenkov model gc --dry-run
cherenkov model gc
cherenkov model remove small-moe
```

`--source-only` releases a retained source copy after checking the prepared
weights and configuration. Plain `remove` removes the index entry.
Neither command deletes files. `gc` collects unreferenced owned directories
and abandoned imports. External
directories are left in place. Live leases protect stores in use by readers or
imports. These commands also accept `--json`.

Reported bytes are file lengths, not allocated disk blocks. Hard-linked files
can appear in more than one store's total. External-byte totals cover registered
artifact directories, excluding raw local sources.

## Storage and interfaces

```text
data/
  index.json           model identities, sources and artifact references
  index.lock           catalog lock
  artifacts/<id>/      owned prepared stores or retained source downloads
  leases/<id>          reader/import locks
```

The catalog is replaced atomically. Ownership is recorded explicitly; a local
folder does not become owned because it lies under the data root. Lock files
remain after collection so concurrent processes use the same lock identity.
Registering an existing managed entry reuses its ownership record. Other paths
inside managed artifacts, including private HF snapshot paths, are rejected.
See [storage](storage.md) for root selection.

`ModelIndex` manages registration, resolution, packing, and collection.
`ArtifactLease` keeps a resolved directory live while its byte sources, mappings,
and GPU views are used. The `cherenkov-model-data` crate describes containers,
tensor encodings, byte reads, and optional mappings without Metal dependencies.
HF header discovery implements that same byte-source interface.

`StoreDiscovery` adapters advertise search and enumeration separately, with their
own typed filters such as HF's `author`. The `Discovery` dispatcher validates
requests before calling an adapter and attaches a stable store ID to each result.
Continuation cursors belong to that store and request, including its page size.
Results carry optional common metadata and provider-specific fields. Adapters
exclude unknown values from filter matches and report known metadata gaps.
Search adapters and CLI search commands are not yet implemented.

The current backend manages complete directories. Cross-model object deduplication,
independent n-gram artifacts, and shard-at-a-time payload
conversion remain future work.
