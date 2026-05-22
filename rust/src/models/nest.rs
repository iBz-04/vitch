use tch::{Kind, Tensor, nn};

use crate::{
    config::NesTConfig,
    conv_layers::ChannelLayerNorm2D,
    tensor::{assert_image_patchable, patch_dim},
};

fn repeat_block_repeats(repeats: &[usize], num_hierarchies: usize) -> Vec<usize> {
    match repeats.len() {
        0 => panic!("block repeats must not be empty"),
        1 => vec![repeats[0]; num_hierarchies],
        len if len == num_hierarchies => repeats.to_vec(),
        _ => panic!("block repeats must have length 1 or match num hierarchies"),
    }
}

#[derive(Debug)]
struct PatchEmbedding {
    patch_size: i64,
    norm1: ChannelLayerNorm2D,
    projection: nn::Conv2D,
    norm2: ChannelLayerNorm2D,
}

impl PatchEmbedding {
    fn new(vs: &nn::Path, patch_size: i64, channels: i64, dim_out: i64) -> Self {
        assert!(patch_size > 0, "patch size must be positive");
        let patch_dim = patch_dim(channels, patch_size, patch_size);
        let norm1 = ChannelLayerNorm2D::new(&(vs / "norm1"), patch_dim, 1e-5);
        let projection = nn::conv2d(vs / "projection", patch_dim, dim_out, 1, Default::default());
        let norm2 = ChannelLayerNorm2D::new(&(vs / "norm2"), dim_out, 1e-5);

        Self {
            patch_size,
            norm1,
            projection,
            norm2,
        }
    }
}

impl nn::Module for PatchEmbedding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let size = xs.size();
        assert_eq!(
            size.len(),
            4,
            "expected image tensor with shape [batch, channels, height, width]"
        );
        let batch = size[0];
        let channels = size[1];
        let height = size[2];
        let width = size[3];
        assert_image_patchable(height, width, self.patch_size, self.patch_size);
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;

        xs.view([
            batch,
            channels,
            grid_h,
            self.patch_size,
            grid_w,
            self.patch_size,
        ])
        .permute([0, 3, 5, 1, 2, 4])
        .contiguous()
        .view([
            batch,
            channels * self.patch_size * self.patch_size,
            grid_h,
            grid_w,
        ])
        .apply(&self.norm1)
        .apply(&self.projection)
        .apply(&self.norm2)
    }
}

#[derive(Debug)]
struct FeedForward {
    norm: ChannelLayerNorm2D,
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    dropout: f64,
}

impl FeedForward {
    fn new(vs: &nn::Path, dim: i64, mlp_mult: i64, dropout: f64) -> Self {
        assert!(mlp_mult > 0, "mlp multiplier must be positive");
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), dim, 1e-5);
        let conv1 = nn::conv2d(vs / "conv1", dim, dim * mlp_mult, 1, Default::default());
        let conv2 = nn::conv2d(vs / "conv2", dim * mlp_mult, dim, 1, Default::default());

        Self {
            norm,
            conv1,
            conv2,
            dropout,
        }
    }
}

impl nn::ModuleT for FeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.norm)
            .apply(&self.conv1)
            .gelu("none")
            .dropout(self.dropout, train)
            .apply(&self.conv2)
            .dropout(self.dropout, train)
    }
}

#[derive(Debug)]
struct Attention {
    norm: ChannelLayerNorm2D,
    to_qkv: nn::Conv2D,
    to_out: nn::Conv2D,
    heads: i64,
    dim_head: i64,
    scale: f64,
    dropout: f64,
}

impl Attention {
    fn new(vs: &nn::Path, dim: i64, heads: i64, dropout: f64) -> Self {
        assert!(heads > 0, "heads must be positive");
        assert_eq!(dim % heads, 0, "dim must be divisible by heads");
        let dim_head = dim / heads;
        let inner_dim = dim_head * heads;
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), dim, 1e-5);
        let to_qkv = nn::conv2d(
            vs / "to_qkv",
            dim,
            inner_dim * 3,
            1,
            nn::ConvConfig {
                bias: false,
                ..Default::default()
            },
        );
        let to_out = nn::conv2d(vs / "to_out", inner_dim, dim, 1, Default::default());

        Self {
            norm,
            to_qkv,
            to_out,
            heads,
            dim_head,
            scale: (dim_head as f64).powf(-0.5),
            dropout,
        }
    }
}

impl nn::ModuleT for Attention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        let batch = size[0];
        let height = size[2];
        let width = size[3];
        let tokens = height * width;
        let qkv = xs.apply(&self.norm).apply(&self.to_qkv).chunk(3, 1);
        let q = qkv[0]
            .view([batch, self.heads, self.dim_head, tokens])
            .transpose(2, 3);
        let k = qkv[1]
            .view([batch, self.heads, self.dim_head, tokens])
            .transpose(2, 3);
        let v = qkv[2]
            .view([batch, self.heads, self.dim_head, tokens])
            .transpose(2, 3);
        let attn = (q.matmul(&k.transpose(-1, -2)) * self.scale)
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train);

        attn.matmul(&v)
            .transpose(2, 3)
            .contiguous()
            .view([batch, self.heads * self.dim_head, height, width])
            .apply(&self.to_out)
            .dropout(self.dropout, train)
    }
}

#[derive(Debug)]
struct TransformerLayer {
    attention: Attention,
    feed_forward: FeedForward,
}

impl TransformerLayer {
    fn new(vs: &nn::Path, dim: i64, heads: i64, mlp_mult: i64, dropout: f64) -> Self {
        let attention = Attention::new(&(vs / "attention"), dim, heads, dropout);
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), dim, mlp_mult, dropout);

        Self {
            attention,
            feed_forward,
        }
    }
}

impl nn::ModuleT for TransformerLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + self.attention.forward_t(xs, train);
        let ff = self.feed_forward.forward_t(&xs, train);

        xs + ff
    }
}

#[derive(Debug)]
struct Transformer {
    pos_embedding: Tensor,
    layers: Vec<TransformerLayer>,
}

impl Transformer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        seq_len: i64,
        depth: usize,
        heads: i64,
        mlp_mult: i64,
        dropout: f64,
    ) -> Self {
        let pos_embedding = vs.var(
            "pos_embedding",
            &[seq_len],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                TransformerLayer::new(&(vs / name.as_str()), dim, heads, mlp_mult, dropout)
            })
            .collect();

        Self {
            pos_embedding,
            layers,
        }
    }
}

impl nn::ModuleT for Transformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        let height = size[2];
        let width = size[3];
        let pos_embedding = self
            .pos_embedding
            .narrow(0, 0, height * width)
            .view([1, 1, height, width]);
        let mut xs = xs + pos_embedding;

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs
    }
}

#[derive(Debug)]
struct Aggregate {
    conv: nn::Conv2D,
    norm: ChannelLayerNorm2D,
}

impl Aggregate {
    fn new(vs: &nn::Path, dim: i64, dim_out: i64) -> Self {
        let conv = nn::conv2d(
            vs / "conv",
            dim,
            dim_out,
            3,
            nn::ConvConfig {
                padding: 1,
                ..Default::default()
            },
        );
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), dim_out, 1e-5);

        Self { conv, norm }
    }
}

impl nn::Module for Aggregate {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.conv)
            .apply(&self.norm)
            .max_pool2d([3, 3], [2, 2], [1, 1], [1, 1], false)
    }
}

#[derive(Debug)]
struct NesTLayer {
    transformer: Transformer,
    aggregate: Option<Aggregate>,
    block_size: i64,
}

impl NesTLayer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        dim_out: i64,
        seq_len: i64,
        depth: usize,
        heads: i64,
        mlp_mult: i64,
        dropout: f64,
        block_size: i64,
        is_last: bool,
    ) -> Self {
        let transformer = Transformer::new(
            &(vs / "transformer"),
            dim,
            seq_len,
            depth,
            heads,
            mlp_mult,
            dropout,
        );
        let aggregate = if is_last {
            None
        } else {
            Some(Aggregate::new(&(vs / "aggregate"), dim, dim_out))
        };

        Self {
            transformer,
            aggregate,
            block_size,
        }
    }
}

impl nn::ModuleT for NesTLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        let batch = size[0];
        let dim = size[1];
        let height = size[2];
        let width = size[3];
        assert_image_patchable(height, width, self.block_size, self.block_size);
        let block_h = height / self.block_size;
        let block_w = width / self.block_size;
        let mut xs = xs
            .view([
                batch,
                dim,
                self.block_size,
                block_h,
                self.block_size,
                block_w,
            ])
            .permute([0, 2, 4, 1, 3, 5])
            .contiguous()
            .view([
                batch * self.block_size * self.block_size,
                dim,
                block_h,
                block_w,
            ]);
        xs = self.transformer.forward_t(&xs, train);
        xs = xs
            .view([
                batch,
                self.block_size,
                self.block_size,
                dim,
                block_h,
                block_w,
            ])
            .permute([0, 3, 1, 4, 2, 5])
            .contiguous()
            .view([batch, dim, height, width]);

        match &self.aggregate {
            Some(aggregate) => xs.apply(aggregate),
            None => xs,
        }
    }
}

#[derive(Debug)]
pub struct NesT {
    patch_embedding: PatchEmbedding,
    layers: Vec<NesTLayer>,
    head_norm: ChannelLayerNorm2D,
    head: nn::Linear,
}

impl NesT {
    pub fn new(vs: &nn::Path, config: NesTConfig) -> Self {
        assert!(
            config.num_hierarchies > 0,
            "num hierarchies must be positive"
        );
        assert_eq!(
            config.image_size.height, config.image_size.width,
            "NesT expects square image size"
        );
        assert_eq!(
            config.patch_size.height, config.patch_size.width,
            "NesT expects square patch size"
        );
        let image_size = config.image_size.height;
        let patch_size = config.patch_size.height;
        assert_image_patchable(image_size, image_size, patch_size, patch_size);
        let fmap_size = image_size / patch_size;
        let max_blocks = 2_i64.pow((config.num_hierarchies - 1) as u32);
        assert_image_patchable(fmap_size, fmap_size, max_blocks, max_blocks);
        let seq_len = (fmap_size / max_blocks).pow(2);
        let block_repeats = repeat_block_repeats(&config.block_repeats, config.num_hierarchies);
        let mults: Vec<i64> = (0..config.num_hierarchies)
            .rev()
            .map(|level| 2_i64.pow(level as u32))
            .collect();
        let layer_dims: Vec<i64> = mults.iter().map(|mult| config.dim * mult).collect();
        let patch_embedding = PatchEmbedding::new(
            &(vs / "patch_embedding"),
            patch_size,
            config.channels,
            layer_dims[0],
        );
        let mut dims_with_tail = layer_dims.clone();
        dims_with_tail.push(*layer_dims.last().expect("last dim"));
        let layers = (0..config.num_hierarchies)
            .map(|index| {
                let level = config.num_hierarchies - 1 - index;
                let name = format!("layer_{index}");
                NesTLayer::new(
                    &(vs / name.as_str()),
                    dims_with_tail[index],
                    dims_with_tail[index + 1],
                    seq_len,
                    block_repeats[index],
                    config.heads * mults[index],
                    config.mlp_mult,
                    config.dropout,
                    2_i64.pow(level as u32),
                    level == 0,
                )
            })
            .collect();
        let last_dim = *layer_dims.last().expect("last dim");
        let head_norm = ChannelLayerNorm2D::new(&(vs / "head_norm"), last_dim, 1e-5);
        let head = nn::linear(
            vs / "head",
            last_dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            patch_embedding,
            layers,
            head_norm,
            head,
        }
    }
}

impl nn::ModuleT for NesT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.apply(&self.patch_embedding);

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs.apply(&self.head_norm)
            .mean_dim(&[2_i64, 3][..], false, Kind::Float)
            .apply(&self.head)
    }
}
