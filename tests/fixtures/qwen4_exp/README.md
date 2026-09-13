# Qwen4-exp test checkpoint

This BF16 fixture is
[inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B](https://huggingface.co/inference-optimization/Qwen3.8-Flash-Next-0.2B-A0.2B/tree/5cdc1eff790ad299680eda7b97241068224581e4)
at revision `5cdc1eff790ad299680eda7b97241068224581e4` (MIT).
It has four decoder layers and no MTP weights. The weights and tokenizer
are stored in Git LFS; the other files are ordinary Git files.

```sh
mise install
mise exec -- git lfs install --local
mise exec -- git lfs pull
cargo test --release --test import -- --ignored --test-threads=1
```

The tests pack into temporary directories, compare Metal prefill and decode
with the CPU reference, and run CLI generation. They also check mapped tensor
views and indexed-store ownership through source removal, variant replacement,
and garbage collection. The full fixture suite requires Apple Silicon.
The default unit suite does not load it.

To clone without fetching the fixture, set `GIT_LFS_SKIP_SMUDGE=1` for the clone.
