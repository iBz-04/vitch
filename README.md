# vit-tch

This repository is being rewritten as a Rust implementation of Vision Transformer models using [`tch`], the Rust bindings for LibTorch.

The Python implementation remains useful as the reference while the Rust crate grows, but new implementation work should happen in `rust/`.

## Current Scope

The Rust crate currently covers the clean foundation for the port:

- Typed model config structs
- Tensor helpers for image, sequence, and video patching
- Shared transformer layers
- Baseline `ViT`, `SimpleViT`, `ViT1D`, `ViT3D`, and small variants
- `ViTWithDecorr`, `MAE`, and `SimMIM`
- Shape tests and smoke examples

## Porting Direction

The port should continue in small, reviewable phases:

1. Keep the core Rust crate compiling and tested.
2. Add low-risk model variants that reuse existing primitives.
3. Extract shared helpers before introducing larger architecture families.
4. Add focused tests for every ported public model.
5. Avoid large monolithic files and one-off fixes.

High-risk features such as nested tensors, dynamic token routing, flash attention behavior, FFTs, and audio preprocessing should wait until the core crate remains stable across several model families.

## Next Phase Checklist

- Port compact model families one module at a time, starting with variants that reuse current transformer and tensor primitives.
- Keep configs typed and explicit instead of passing loose option maps.
- Wire every public model through `config.rs`, `models/mod.rs`, `lib.rs`, tests, and at least one small example when useful.
- Prefer shared helpers for reusable behavior, but avoid broad abstractions until two or more models need the same logic.
- Validate with `cargo fmt`, `cargo check --tests --examples`, and focused shape or smoke coverage before moving on.

## Rust Crate

See `rust/README.md` for build, test, and example commands.

## Validation Standard

Every ported model should have at least one shape test. Training-oriented wrappers should also produce a scalar loss and type-check through examples or smoke tests.
