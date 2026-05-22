# vit_tch

Rust Vision Transformer models built on `tch`.

## Build

```bash
cargo check
cargo check --features target-checks --tests --examples
```

The default feature uses `tch/doc-only`, which is suitable for type-checking in environments without a local LibTorch installation. Use `cargo check --features target-checks --tests --examples` for lightweight validation of test and example targets; runnable tests need the runtime feature and a LibTorch build.

For a runtime build, provide LibTorch with `LIBTORCH` or `LIBTORCH_USE_PYTORCH=1`, then run without the default `doc-only` feature:

```bash
cargo run --no-default-features --features runtime --example vit_inference
```

The `download-libtorch` feature is also available as a shortcut when `tch` can download a compatible LibTorch package for the platform.

## Examples

```bash
cargo run --no-default-features --features runtime --example vit_inference
cargo run --no-default-features --features runtime --example cct_inference
cargo run --no-default-features --features runtime --example cvt_inference
cargo run --no-default-features --features runtime --example pit_inference
cargo run --no-default-features --features runtime --example levit_inference
cargo run --no-default-features --features runtime --example mobile_vit_inference
cargo run --no-default-features --features runtime --example vit_1d
cargo run --no-default-features --features runtime --example vit_3d
cargo run --no-default-features --features runtime --example vivit_inference
cargo run --no-default-features --features runtime --example vit_for_small_dataset
cargo run --no-default-features --features runtime --example decorr_smoke_train
cargo run --no-default-features --features runtime --example mae_smoke_train
cargo run --no-default-features --features runtime --example simmim_smoke_train
cargo run --no-default-features --features runtime --example distill_smoke_train
```

## Tests

```bash
cargo test --no-default-features --features runtime
```

Full tests require a usable LibTorch runtime. You can provide one with `LIBTORCH`, with `LIBTORCH_USE_PYTORCH=1` when Python has the matching PyTorch version installed, or with `download-libtorch` when the platform package includes the full LibTorch headers and libraries. In lightweight CI or local environments without LibTorch, use:

```bash
cargo check --features target-checks --tests --examples
```

On macOS with Miniforge/Conda, `torch-sys 0.24.0` expects PyTorch `2.11.0`. Install a matching Python package and expose PyTorch's bundled dylibs before running runtime tests:

```bash
python -m pip install --force-reinstall "torch==2.11.0" "torchvision==0.26.0"
TORCH_LIB_DIR="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')"
VIRTUAL_ENV="$CONDA_PREFIX" LIBTORCH_USE_PYTORCH=1 DYLD_LIBRARY_PATH="$TORCH_LIB_DIR:${DYLD_LIBRARY_PATH:-}" cargo test --manifest-path rust/Cargo.toml --no-default-features --features runtime
```

## Public Surface

The crate exposes typed configs and model structs from the root module:

```rust
use vit_tch::{ImageSize, ViT, ViTConfig};
```

Keep new model ports in focused modules under `src/models/`, and add shape or smoke coverage in `tests/`.
