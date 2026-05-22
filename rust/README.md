# vit_tch

Rust Vision Transformer models built on `tch`.

## Build

```bash
cargo check
cargo check --tests --examples
```

The default feature uses `tch/doc-only`, which is suitable for type-checking in environments without a local LibTorch installation.

For a runtime build with LibTorch download support:

```bash
cargo run --no-default-features --features download-libtorch --example vit_inference
```

## Examples

```bash
cargo run --example vit_inference
cargo run --example cct_inference
cargo run --example cvt_inference
cargo run --example vit_1d
cargo run --example vit_3d
cargo run --example vit_for_small_dataset
cargo run --example decorr_smoke_train
cargo run --example mae_smoke_train
cargo run --example simmim_smoke_train
```

## Tests

```bash
cargo test
```

Full tests require a usable LibTorch runtime. In lightweight CI or local environments without LibTorch, use:

```bash
cargo check --tests --examples
```

## Public Surface

The crate exposes typed configs and model structs from the root module:

```rust
use vit_tch::{ImageSize, ViT, ViTConfig};
```

Keep new model ports in focused modules under `src/models/`, and add shape or smoke coverage in `tests/`.
