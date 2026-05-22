# vit-tch

Rust Vision Transformer models built on [`tch`], the Rust bindings for LibTorch.

This repository is Rust-only. The crate lives in [`rust/`](rust/) and exposes typed configs, model structs, examples, and runtime-ready shape/smoke tests.

## Build

```bash
cargo check --manifest-path rust/Cargo.toml --features target-checks --tests --examples
```

The default crate feature uses `tch/doc-only` for lightweight type-checking without a local LibTorch runtime. Runnable tests and examples need LibTorch through `LIBTORCH`, `LIBTORCH_USE_PYTORCH=1`, or the crate's `download-libtorch` feature.

## Runtime Tests

```bash
cargo test --manifest-path rust/Cargo.toml --no-default-features --features runtime
```

On macOS with Miniforge/Conda and PyTorch-provided LibTorch:

```bash
python -m pip install --force-reinstall "torch==2.11.0" "torchvision==0.26.0"
TORCH_LIB_DIR="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')"
VIRTUAL_ENV="$CONDA_PREFIX" LIBTORCH_USE_PYTORCH=1 DYLD_LIBRARY_PATH="$TORCH_LIB_DIR:${DYLD_LIBRARY_PATH:-}" cargo test --manifest-path rust/Cargo.toml --no-default-features --features runtime
```

## Documentation

See [`rust/README.md`](rust/README.md) for the public surface, examples, feature flags, and validation commands.
