# Intro

Vision Transformer models in Rust, built with [`tch`], the Rust bindings for LibTorch.

This crate is for building and experimenting with ViT-style image, video, sequence, and self-supervised transformer models in Rust. It provides typed configs, reusable model structs, runnable examples, and shape tests for research prototypes and Rust deep learning projects.

Below you will see its working principle both visually and mathematically.



### Overview

![Vision Transformer](images/vit.gif)

The image is split into fixed-size patches, projected into tokens, processed through transformer layers, and pooled into a final prediction.


## How it works

A Vision Transformer treats an image like a sequence.

Normal images have this shape:

```text
[batch, channels, height, width]
```

The model changes the image into this shape:

```text
[batch, tokens, dim]
```

The flow is:

1. Split the image into patches.
2. Flatten each patch into one long vector.
3. Project each patch vector into `dim`.
4. Add position embeddings.
5. Run transformer layers.
6. Pool the tokens.
7. Predict class logits.

## Patch Math

For an image with shape `C x H x W` and patch size `P x P`:

```text
number_of_patches = (H / P) * (W / P)
patch_dim = C * P * P
```

Example:

```text
image = 3 x 256 x 256
patch = 32 x 32
number_of_patches = (256 / 32) * (256 / 32) = 64
patch_dim = 3 * 32 * 32 = 3072
```

So one image becomes 64 patch tokens. Each token starts with 3072 values, then the model maps it into the hidden size `dim`.

## Position Embeddings

After patching, the model has a list of tokens. A list alone does not tell the model where each patch came from.

Position embeddings fix that.

```text
token = patch_embedding + position_embedding
```

This lets the model know that one patch came from the top-left, another from the center, and another from the bottom-right.

## Class Token And Pooling

The model can summarize the image in two ways.

`Pool::Cls` adds a learned class token. This token gathers information from all patch tokens. The final class token is used for prediction.

`Pool::Mean` does not add a class token. It averages all final patch tokens and uses that average for prediction.

Both methods produce one vector for the whole image.

## Self-Attention Math

Self-attention is the main science behind ViT.

Each token becomes three vectors:

```text
Q = query
K = key
V = value
```

The model compares every query with every key:

```text
scores = QK^T / sqrt(d)
weights = softmax(scores)
output = weights V
```

In simple words:

- `QK^T` measures how much each patch should care about each other patch.
- `sqrt(d)` keeps the scores from getting too large.
- `softmax` turns scores into weights that add up to 1.
- `weights V` mixes useful information from other patches.

This is why one patch can use information from the whole image.

## Multi-Head Attention

One attention head learns one kind of relation.

Many heads run at the same time. One head may focus on edges. Another may focus on shape. Another may focus on object parts.

The heads are joined together and projected back into the model dimension.

```text
heads = split(tokens)
attended = attention(heads)
tokens = merge(attended)
```

## Transformer Block

A transformer block has two main parts:

```text
tokens -> attention -> tokens -> feed_forward -> tokens
```

Attention mixes information between patches.

The feed-forward layer improves each token on its own.

Layer normalization keeps training stable. Dropout can help reduce overfitting.

## Output

The last step is the prediction head.

```text
image_vector -> linear_layer -> class_logits
```

`class_logits` are raw scores. The highest score is the predicted class.

For `num_classes = 1000`, one image gives:

```text
[1, 1000]
```

For a batch of 8 images:

```text
[8, 1000]
```

### Model Architectures

Each model adapts or extends the core ViT idea for a different goal or constraint.

#### CaiT
Class-Attention in Image Transformers. Separates patch and class token processing into two stages: patch self-attention first, then class-to-patch cross-attention. This lets the model go deeper without instability.

![CaiT](images/cait.png)

#### CvT
Convolutional Vision Transformer. Introduces convolutional projections into token embedding and attention, combining the local inductive bias of convolutions with global transformer attention.

![CvT](images/cvt.png)

#### PiT
Pooling-based Image Transformer. Applies spatial pooling to reduce the token sequence across depth, similar to how CNNs downsample feature maps.

![PiT](images/pit.png)

#### LeViT
A hybrid architecture for fast inference. Uses a convolutional stem followed by strided attention to progressively shrink spatial resolution, making it suitable for edge deployment.

![LeViT](images/levit.png)

#### MaxViT
Multi-Axis Vision Transformer. Alternates between local window attention and dilated grid attention, giving both fine-grained local and global coverage without quadratic cost.

![MaxViT](images/max-vit.png)

#### MAE
Masked Autoencoder. Trains the encoder by masking a large fraction of patches and asking the decoder to reconstruct the missing pixel values. Learns strong representations without labels.

![MAE](images/mae.png)

#### SimMIM
Simple Masked Image Modeling. Predicts raw pixel values for randomly masked patches using a single linear head, showing that simple reconstruction objectives are effective.

![SimMIM](images/simmim.png)

#### DINO
Self-supervised training with self-distillation and no labels. A student network is trained to match a momentum teacher output, producing features useful for segmentation and retrieval.

![DINO](images/dino.png)

#### ViViT
Video Vision Transformer. Extends ViT to video by factorizing spatial and temporal attention across frames.

![ViViT](images/vivit.png)

#### MobileViT
A lightweight ViT designed for mobile devices. Combines depthwise convolutions with local transformer blocks to reduce compute while keeping accuracy.

![MobileViT](images/mbvit.png)

#### Distillation
Knowledge distillation for ViT. Introduces a distillation token alongside the class token, trained to match the output of a teacher network such as a CNN.

![Distillation](images/distill.png)

#### T2T ViT
Tokens-to-Token ViT. Aggregates neighboring tokens into one token across layers, progressively reducing sequence length while building richer representations.

![T2T ViT](images/t2t.png)

#### CrossViT
Processes images at two different patch scales in parallel branches, then fuses information between them via cross-attention.

![CrossViT](images/cross_vit.png)

#### CrossFormer
Uses cross-scale attention between tokens at different granularities, connecting coarse and fine-grained features across the hierarchy.

![CrossFormer](images/crossformer.png)

![CrossFormer Detail](images/crossformer2.png)

#### ViT for Small Datasets
Modifications to ViT that improve performance when training data is limited, using shifted patch tokenization and locality self-attention.

![ViT for Small Datasets](images/vit_for_small_datasets.png)

#### ATS
Adaptive Token Sampling. Dynamically selects the most informative tokens at each layer, reducing computation by dropping less useful patches.

![ATS](images/ats.png)

#### PatchMerger
Learns to merge patch tokens mid-network, compressing the sequence length at a fixed point to reduce the cost of later layers.

![PatchMerger](images/patch_merger.png)

#### NesT
Nested ViT. Processes tokens in local blocks at each stage then aggregates them hierarchically, building a pyramid of representations similar to CNNs.

![NesT](images/nest.png)

#### RegionViT
Introduces regional tokens that summarize groups of local tokens, enabling efficient global communication without attending over all patches.

![RegionViT](images/regionvit.png)

![RegionViT Detail](images/regionvit2.png)

#### Scalable ViT
Uses interactive window attention that scales to high resolution by splitting attention into local windows and a global context channel.

![Scalable ViT](images/scalable-vit-1.png)

![Scalable ViT Detail](images/scalable-vit-2.png)

#### SepViT
Separable ViT. Applies depthwise separable attention, factorizing the attention operation to reduce parameters and FLOPs.

![SepViT](images/sep-vit.png)

#### EsViT
Efficient Self-supervised ViT. Extends DINO with a multi-stage architecture and region-level matching for more efficient self-supervised pretraining.

![EsViT](images/esvit.png)

#### Parallel ViT
Runs multiple attention and MLP blocks in parallel rather than sequentially, then sums their outputs. Improves throughput while maintaining accuracy.

![Parallel ViT](images/parallel-vit.png)

#### NaViT
Native Resolution ViT. Removes the fixed resolution constraint by packing multiple images of varying sizes into a single sequence using fractional positional embeddings.

![NaViT](images/navit.png)

#### MP3
Multi-patch prediction pretraining. Predicts multiple masked patch regions simultaneously, improving the density and quality of the self-supervised signal.

![MP3](images/mp3.png)

#### Learnable Memory ViT
Appends a set of learnable memory tokens to each layer's key-value pairs, giving the model access to persistent task-specific context without increasing sequence length.

![Learnable Memory ViT](images/learnable-memory-vit.png)

#### Twins SVT
Twin Transformers with Spatially Separable self-attention. Alternates between local grouped attention and global strided attention to cover both fine and coarse structure efficiently.

![Twins SVT](images/twins_svt.png)

#### XCiT
Cross-Covariance Image Transformer. Transposes attention to operate on the feature dimension rather than the token dimension, achieving linear complexity with respect to the number of tokens.

![XCiT](images/xcit.png)


## What This Project Implements

This crate implements Vision Transformer building blocks in Rust:

- patch embedding
- position embedding
- class token and mean pooling
- multi-head self-attention
- feed-forward layers
- transformer blocks
- image, video, and sequence variants
- runtime examples and shape tests

The public API is exported from the crate root:

```rust
use vit_tch::{ImageSize, ViT, ViTConfig};
```

## Example

```rust
use tch::{Device, Kind, Tensor, nn, nn::ModuleT};
use vit_tch::{ImageSize, ViT, ViTConfig};

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(256),
            patch_size: ImageSize::square(32),
            num_classes: 1000,
            dim: 1024,
            depth: 6,
            heads: 16,
            mlp_dim: 2048,
            ..Default::default()
        },
    );

    let img = Tensor::randn([1, 3, 256, 256], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    println!("{:?}", logits.size());
}
```

Expected output shape:

```text
[1, 1000]
```

## Main Config Fields

`image_size` is the input image size.

`patch_size` is the size of each image patch.

`num_classes` is the number of output classes.

`dim` is the hidden token size.

`depth` is the number of transformer layers.

`heads` is the number of attention heads.

`mlp_dim` is the hidden size inside the feed-forward layer.

`dropout` and `emb_dropout` control dropout.

## Supported Models

These models share the same idea: turn input into tokens, process tokens, then produce an output.

| Model | Use case |
| --- | --- |
| `ViT` | Base Vision Transformer for image inputs |
| `SimpleViT` | Minimal ViT variant |
| `DeepViT` | Deeper ViT-style architecture |
| `LocalViT` | ViT with local feature mixing |
| `CaiT` | Class-attention image transformer |
| `CCT` | Compact convolutional transformer |
| `CvT` | Convolutional Vision Transformer |
| `CrossFormer` | Cross-scale image transformer |
| `LeViT` | Fast hybrid transformer for efficient inference |
| `MaxViT` | Multi-axis local and global attention |
| `MobileViT` | Lightweight mobile-oriented vision transformer |
| `NesT` | Nested hierarchical vision transformer |
| `PiT` | Pooling-based vision transformer |
| `RegionViT` | Regional-token vision transformer |
| `SepViT` | Separable attention vision transformer |
| `TwinsSVT` | Spatially separable hierarchical transformer |
| `XCiT` | Cross-covariance image transformer |
| `ViTForSmallDataset` | ViT variant for smaller training datasets |
| `ViTWithDecorr` | ViT with decorrelation auxiliary loss |
| `ViTWithPatchDropout` | ViT with patch dropout |
| `SimpleViTWithPatchDropout` | SimpleViT with patch dropout |
| `SimpleViTWithQkNorm` | SimpleViT with query-key normalization |
| `SimpleViTWithRegisterTokens` | SimpleViT with register tokens |
| `ViT1D` | ViT for 1D sequence inputs |
| `SimpleViT1D` | Simple ViT variant for 1D sequence inputs |
| `ViT3D` | ViT for 3D inputs |
| `SimpleViT3D` | Simple ViT variant for 3D inputs |
| `ViViT` | Video Vision Transformer |
| `AcceptVideoWrapper` | Wrapper for video-style inputs |
| `MAE` | Masked autoencoder pretraining |
| `SimMIM` | Simple masked image modeling |
| `DINO` | Self-supervised distillation |
| `Distill` | Teacher-student distillation |
| `MP3` | Multi-patch prediction pretraining |
| `MPP` | Masked patch prediction |
| `NaViT` | Native-resolution ViT |
| `NaViTNestedTensor` | NaViT variant using nested tensors |
| `ATS` | Adaptive token sampling |
| `VAT` | Token/attention adaptation variant |
| `VAAT` | Attention adaptation variant |
| `ParallelViT` | Parallel attention/MLP ViT variant |

## Build

Run type checks:

```bash
cargo check --features target-checks --tests --examples
```

Set up the runtime once:

```bash
scripts/setup-runtime.sh
```

Run a basic example:

```bash
scripts/run-example.sh vit_inference
```

Run runtime tests:

```bash
source .env.runtime
cargo test --no-default-features --features runtime
```

The default feature uses `tch/doc-only`, so type-checking does not need a local LibTorch runtime.

This crate pins `tch` to a release that matches PyTorch `2.7.0`, because `tch` links against the PyTorch C++ API and must use the same library version at build and runtime.

Running examples and runtime tests needs LibTorch through one of these:

- `scripts/setup-runtime.sh`, which creates `.venv`, installs the matching PyTorch wheel, and writes `.env.runtime`
- `LIBTORCH`, if you already have a local LibTorch or PyTorch install
- `LIBTORCH_USE_PYTORCH=1`, if your active Python environment already has the matching PyTorch version installed
- `download-libtorch`, where prebuilt LibTorch archives are available

## Why ViT Matters

A convolution model usually builds vision from nearby pixels first.

A Vision Transformer can connect any patch to any other patch in one step.

That makes it good at learning:

- local details
- object parts
- long-range shape
- whole-scene context

The main tradeoff is data and compute. ViT models often need more training data or stronger training methods than small convolution models.
