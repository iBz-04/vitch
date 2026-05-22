use tch::{IndexOp, Kind, Tensor, nn, nn::ModuleT};

use crate::{
    config::RegionViTConfig,
    conv_layers::ChannelLayerNorm2D,
    layers::FeedForward,
    tensor::{assert_image_patchable, merge_heads, patch_dim, split_heads},
};

#[derive(Debug)]
struct Downsample {
    conv: nn::Conv2D,
}

impl Downsample {
    fn new(vs: &nn::Path, dim_in: i64, dim_out: i64) -> Self {
        let conv = nn::conv2d(
            vs / "conv",
            dim_in,
            dim_out,
            3,
            nn::ConvConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
        );

        Self { conv }
    }
}

impl nn::Module for Downsample {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.conv)
    }
}

#[derive(Debug)]
struct PEG {
    proj: nn::Conv2D,
}

impl PEG {
    fn new(vs: &nn::Path, dim: i64, kernel_size: i64) -> Self {
        assert!(kernel_size > 0, "peg kernel size must be positive");
        let proj = nn::conv2d(
            vs / "proj",
            dim,
            dim,
            kernel_size,
            nn::ConvConfig {
                padding: kernel_size / 2,
                groups: dim,
                ..Default::default()
            },
        );

        Self { proj }
    }
}

impl nn::Module for PEG {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs + xs.apply(&self.proj)
    }
}

#[derive(Debug)]
enum LocalEncoder {
    Conv(nn::Conv2D),
    ThreeConv {
        conv1: nn::Conv2D,
        norm1: ChannelLayerNorm2D,
        conv2: nn::Conv2D,
        norm2: ChannelLayerNorm2D,
        conv3: nn::Conv2D,
    },
}

impl LocalEncoder {
    fn new(vs: &nn::Path, channels: i64, dim: i64, tokenize_local_3_conv: bool) -> Self {
        if tokenize_local_3_conv {
            let conv1 = nn::conv2d(
                vs / "conv1",
                channels,
                dim,
                3,
                nn::ConvConfig {
                    stride: 2,
                    padding: 1,
                    ..Default::default()
                },
            );
            let norm1 = ChannelLayerNorm2D::new(&(vs / "norm1"), dim, 1e-5);
            let conv2 = nn::conv2d(
                vs / "conv2",
                dim,
                dim,
                3,
                nn::ConvConfig {
                    stride: 2,
                    padding: 1,
                    ..Default::default()
                },
            );
            let norm2 = ChannelLayerNorm2D::new(&(vs / "norm2"), dim, 1e-5);
            let conv3 = nn::conv2d(
                vs / "conv3",
                dim,
                dim,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            );

            Self::ThreeConv {
                conv1,
                norm1,
                conv2,
                norm2,
                conv3,
            }
        } else {
            Self::Conv(nn::conv2d(
                vs / "conv",
                channels,
                dim,
                8,
                nn::ConvConfig {
                    stride: 4,
                    padding: 3,
                    ..Default::default()
                },
            ))
        }
    }
}

impl nn::Module for LocalEncoder {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match self {
            Self::Conv(conv) => xs.apply(conv),
            Self::ThreeConv {
                conv1,
                norm1,
                conv2,
                norm2,
                conv3,
            } => xs
                .apply(conv1)
                .apply(norm1)
                .gelu("none")
                .apply(conv2)
                .apply(norm2)
                .gelu("none")
                .apply(conv3),
        }
    }
}

#[derive(Debug)]
struct RegionEncoder {
    patch_size: i64,
    conv: nn::Conv2D,
}

impl RegionEncoder {
    fn new(vs: &nn::Path, channels: i64, dim: i64, patch_size: i64) -> Self {
        assert!(patch_size > 0, "region patch size must be positive");
        let conv = nn::conv2d(
            vs / "conv",
            patch_dim(channels, patch_size, patch_size),
            dim,
            1,
            Default::default(),
        );

        Self { patch_size, conv }
    }
}

impl nn::Module for RegionEncoder {
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
        .permute([0, 1, 3, 5, 2, 4])
        .contiguous()
        .view([
            batch,
            channels * self.patch_size * self.patch_size,
            grid_h,
            grid_w,
        ])
        .apply(&self.conv)
    }
}

#[derive(Debug)]
struct Attention {
    norm: nn::LayerNorm,
    to_qkv: nn::Linear,
    to_out: nn::Linear,
    heads: i64,
    scale: f64,
    dropout: f64,
}

impl Attention {
    fn new(vs: &nn::Path, dim: i64, heads: i64, dim_head: i64, dropout: f64) -> Self {
        assert!(heads > 0, "heads must be positive");
        assert!(dim_head > 0, "dim head must be positive");
        let inner_dim = heads * dim_head;
        let norm = nn::layer_norm(vs / "norm", vec![dim], Default::default());
        let to_qkv = nn::linear(
            vs / "to_qkv",
            dim,
            inner_dim * 3,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let to_out = nn::linear(vs / "to_out", inner_dim, dim, Default::default());

        Self {
            norm,
            to_qkv,
            to_out,
            heads,
            scale: (dim_head as f64).powf(-0.5),
            dropout,
        }
    }

    fn forward_t_with_bias(&self, xs: &Tensor, bias: Option<&Tensor>, train: bool) -> Tensor {
        let qkv = xs.apply(&self.norm).apply(&self.to_qkv).chunk(3, -1);
        let q = split_heads(&qkv[0], self.heads) * self.scale;
        let k = split_heads(&qkv[1], self.heads);
        let v = split_heads(&qkv[2], self.heads);
        let mut dots = q.matmul(&k.transpose(-1, -2));

        if let Some(bias) = bias {
            dots += bias;
        }

        let attn = dots.softmax(-1, Kind::Float).dropout(self.dropout, train);

        merge_heads(&attn.matmul(&v))
            .apply(&self.to_out)
            .dropout(self.dropout, train)
    }
}

impl nn::ModuleT for Attention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward_t_with_bias(xs, None, train)
    }
}

#[derive(Debug)]
struct R2LLayer {
    attention: Attention,
    feed_forward: FeedForward,
}

impl R2LLayer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        heads: i64,
        dim_head: i64,
        ff_dropout: f64,
        attn_dropout: f64,
    ) -> Self {
        let attention = Attention::new(&(vs / "attention"), dim, heads, dim_head, attn_dropout);
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), dim, dim * 4, ff_dropout);

        Self {
            attention,
            feed_forward,
        }
    }
}

#[derive(Debug)]
struct R2LTransformer {
    layers: Vec<R2LLayer>,
    local_rel_pos_bias: Tensor,
    window_size: i64,
    heads: i64,
}

impl R2LTransformer {
    fn new(vs: &nn::Path, dim: i64, config: &RegionViTConfig, depth: usize) -> Self {
        let rel_positions = 2 * config.window_size - 1;
        let local_rel_pos_bias = vs.var(
            "local_rel_pos_bias",
            &[rel_positions * rel_positions, config.heads],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                R2LLayer::new(
                    &(vs / name.as_str()),
                    dim,
                    config.heads,
                    config.dim_head,
                    config.ff_dropout,
                    config.attn_dropout,
                )
            })
            .collect();

        Self {
            layers,
            local_rel_pos_bias,
            window_size: config.window_size,
            heads: config.heads,
        }
    }

    fn relative_position_bias(&self, window_h: i64, window_w: i64) -> Tensor {
        let positions = (0..window_h)
            .flat_map(|h| (0..window_w).map(move |w| (h, w)))
            .collect::<Vec<_>>();
        let mut indices = Vec::with_capacity(positions.len() * positions.len());
        let span = self.window_size * 2 - 1;

        for &(h1, w1) in &positions {
            for &(h2, w2) in &positions {
                let rel_h = h1 - h2 + self.window_size - 1;
                let rel_w = w1 - w2 + self.window_size - 1;
                indices.push(rel_h * span + rel_w);
            }
        }

        let tokens = window_h * window_w;
        let index = Tensor::from_slice(&indices)
            .to_kind(Kind::Int64)
            .to_device(self.local_rel_pos_bias.device());
        let bias = self
            .local_rel_pos_bias
            .index_select(0, &index)
            .view([tokens, tokens, self.heads])
            .permute([2, 0, 1])
            .unsqueeze(0);
        let zeros = Tensor::zeros(
            [1, self.heads, tokens + 1, tokens + 1],
            (bias.kind(), bias.device()),
        );

        zeros.i((.., .., 1.., 1..)).copy_(&bias);
        zeros
    }

    fn forward_pair_t(
        &self,
        local_tokens: &Tensor,
        region_tokens: &Tensor,
        train: bool,
    ) -> (Tensor, Tensor) {
        let local_size = local_tokens.size();
        let region_size = region_tokens.size();
        let batch = local_size[0];
        let dim = local_size[1];
        let local_h = local_size[2];
        let local_w = local_size[3];
        let region_h = region_size[2];
        let region_w = region_size[3];
        assert_image_patchable(local_h, local_w, region_h, region_w);
        let window_h = local_h / region_h;
        let window_w = local_w / region_w;
        let local_window_tokens = window_h * window_w;
        let rel_pos_bias = self.relative_position_bias(window_h, window_w);
        let mut local_tokens =
            local_tokens
                .permute([0, 2, 3, 1])
                .contiguous()
                .view([batch, local_h * local_w, dim]);
        let mut region_tokens = region_tokens.permute([0, 2, 3, 1]).contiguous().view([
            batch,
            region_h * region_w,
            dim,
        ]);

        for layer in &self.layers {
            let region_attn = layer.attention.forward_t(&region_tokens, train);
            region_tokens = &region_tokens + region_attn;
            let local_windows = local_tokens
                .view([batch, local_h, local_w, dim])
                .view([batch, region_h, window_h, region_w, window_w, dim])
                .permute([0, 1, 3, 2, 4, 5])
                .contiguous()
                .view([batch * region_h * region_w, local_window_tokens, dim]);
            let region_windows = region_tokens.view([batch * region_h * region_w, 1, dim]);
            let mut tokens = Tensor::cat(&[region_windows, local_windows], 1);
            let attn = layer
                .attention
                .forward_t_with_bias(&tokens, Some(&rel_pos_bias), train);
            tokens += attn;
            let ff = layer.feed_forward.forward_t(&tokens, train);
            tokens += ff;
            region_tokens = tokens
                .i((.., 0..1, ..))
                .view([batch, region_h * region_w, dim]);
            local_tokens = tokens
                .i((.., 1.., ..))
                .view([batch, region_h, region_w, window_h, window_w, dim])
                .permute([0, 1, 3, 2, 4, 5])
                .contiguous()
                .view([batch, local_h * local_w, dim]);
        }

        let local_tokens = local_tokens
            .view([batch, local_h, local_w, dim])
            .permute([0, 3, 1, 2]);
        let region_tokens = region_tokens
            .view([batch, region_h, region_w, dim])
            .permute([0, 3, 1, 2]);

        (local_tokens, region_tokens)
    }
}

#[derive(Debug)]
struct RegionViTStage {
    downsample: Option<Downsample>,
    peg: Option<PEG>,
    transformer: R2LTransformer,
}

impl RegionViTStage {
    fn new(
        vs: &nn::Path,
        dim_in: i64,
        dim: i64,
        depth: usize,
        config: &RegionViTConfig,
        needs_downsample: bool,
        needs_peg: bool,
    ) -> Self {
        let downsample =
            needs_downsample.then(|| Downsample::new(&(vs / "downsample"), dim_in, dim));
        let peg = needs_peg.then(|| PEG::new(&(vs / "peg"), dim, 3));
        let transformer = R2LTransformer::new(&(vs / "transformer"), dim, config, depth);

        Self {
            downsample,
            peg,
            transformer,
        }
    }

    fn forward_pair_t(
        &self,
        local_tokens: &Tensor,
        region_tokens: &Tensor,
        train: bool,
    ) -> (Tensor, Tensor) {
        let mut local_tokens = local_tokens.shallow_clone();
        let mut region_tokens = region_tokens.shallow_clone();

        if let Some(downsample) = &self.downsample {
            local_tokens = local_tokens.apply(downsample);
            region_tokens = region_tokens.apply(downsample);
        }

        if let Some(peg) = &self.peg {
            local_tokens = local_tokens.apply(peg);
        }

        self.transformer
            .forward_pair_t(&local_tokens, &region_tokens, train)
    }
}

#[derive(Debug)]
pub struct RegionViT {
    local_patch_size: i64,
    region_patch_size: i64,
    local_encoder: LocalEncoder,
    region_encoder: RegionEncoder,
    stages: Vec<RegionViTStage>,
    norm: nn::LayerNorm,
    head: nn::Linear,
}

impl RegionViT {
    pub fn new(vs: &nn::Path, config: RegionViTConfig) -> Self {
        assert!(!config.dim.is_empty(), "dim must not be empty");
        assert_eq!(
            config.depth.len(),
            config.dim.len(),
            "depth must match dim length"
        );
        assert!(
            config.local_patch_size > 0,
            "local patch size must be positive"
        );
        assert!(config.window_size > 0, "window size must be positive");
        let init_dim = config.dim[0];
        let region_patch_size = config.local_patch_size * config.window_size;
        let local_encoder = LocalEncoder::new(
            &(vs / "local_encoder"),
            config.channels,
            init_dim,
            config.tokenize_local_3_conv,
        );
        let region_encoder = RegionEncoder::new(
            &(vs / "region_encoder"),
            config.channels,
            init_dim,
            region_patch_size,
        );
        let mut dim_in = init_dim;
        let stages = config
            .dim
            .iter()
            .zip(&config.depth)
            .enumerate()
            .map(|(index, (&dim, &depth))| {
                let name = format!("stage_{index}");
                let not_first = index != 0;
                let stage = RegionViTStage::new(
                    &(vs / name.as_str()),
                    dim_in,
                    dim,
                    depth,
                    &config,
                    not_first,
                    not_first && config.use_peg,
                );
                dim_in = dim;
                stage
            })
            .collect::<Vec<_>>();
        let last_dim = *config.dim.last().expect("dim must not be empty");
        let norm = nn::layer_norm(vs / "norm", vec![last_dim], Default::default());
        let head = nn::linear(
            vs / "head",
            last_dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            local_patch_size: config.local_patch_size,
            region_patch_size,
            local_encoder,
            region_encoder,
            stages,
            norm,
            head,
        }
    }
}

impl nn::ModuleT for RegionViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        assert_eq!(
            size.len(),
            4,
            "expected image tensor with shape [batch, channels, height, width]"
        );
        assert_image_patchable(
            size[2],
            size[3],
            self.region_patch_size,
            self.region_patch_size,
        );
        assert_image_patchable(
            size[2],
            size[3],
            self.local_patch_size,
            self.local_patch_size,
        );
        let mut local_tokens = xs.apply(&self.local_encoder);
        let mut region_tokens = xs.apply(&self.region_encoder);

        for stage in &self.stages {
            (local_tokens, region_tokens) =
                stage.forward_pair_t(&local_tokens, &region_tokens, train);
        }

        region_tokens
            .mean_dim(&[2_i64, 3][..], false, Kind::Float)
            .apply(&self.norm)
            .apply(&self.head)
    }
}
