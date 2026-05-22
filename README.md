# vit-tch

Vision Transformer models in Rust, built with [`tch`], the Rust bindings for LibTorch.

This crate gives you typed configs, model structs, examples, and shape tests for ViT-style neural networks. It includes the base `ViT` model plus variants for images, video, 1D signals, masked training, distillation, and compact/mobile models.

## Reference Images

Put project images in `images/`.

Useful images for this README:

- `images/patches.*`: an image split into small square patches
- `images/tokens.*`: patches flattened into a token sequence
- `images/attention.*`: attention showing which patches look at each other
- `images/output.*`: the final class prediction

## What ViT Does

A Vision Transformer treats an image like a sentence.

1. Split the image into patches.
2. Turn each patch into a vector.
3. Add position data so the model knows where each patch came from.
4. Pass the patch vectors through transformer layers.
5. Pool the final vectors into one image vector.
6. Use a small head to predict the class.

For an image with shape `C x H x W` and patch size `P x P`:

```text
number_of_patches = (H / P) * (W / P)
patch_dim = C * P * P
```

Each patch becomes one token. The transformer then learns how tokens relate to each other.

## The Math

Self-attention is the main idea.

Each token is projected into three vectors:

```text
Q = query
K = key
V = value
```

Attention compares every query with every key:

```text
attention = softmax((QK^T) / sqrt(d)) V
```

In simple words:

- `QK^T` scores how much each patch should look at every other patch.
- `sqrt(d)` keeps the scores stable.
- `softmax` turns scores into weights.
- Multiplying by `V` mixes information from important patches.

Many attention heads run in parallel. Each head can learn a different kind of visual relation, such as edges, shapes, object parts, or long-range context.

## Why It Works

Convolutions look mostly at nearby pixels. A transformer can connect any patch to any other patch in one step.

That helps the model learn both local details and global structure:

- local: texture, corners, small parts
- global: object shape, layout, scene context

## Quick Use

```rust
use vit_tch::{ImageSize, ViT, ViTConfig};
```

Run a basic example:

```bash
cargo run --no-default-features --features runtime --example vit_inference
```

Run shape checks:

```bash
cargo check --features target-checks --tests --examples
```

Run runtime tests:

```bash
cargo test --no-default-features --features runtime
```

The default feature uses `tch/doc-only`, so type-checking does not need a local LibTorch runtime. Running examples and runtime tests needs LibTorch through `LIBTORCH`, `LIBTORCH_USE_PYTORCH=1`, or the `download-libtorch` feature.
