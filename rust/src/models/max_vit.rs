use tch::{Kind, Tensor, nn};

use crate::{
    config::MaxViTConfig,
    conv_layers::MbConv,
    layers::{Attention, FeedForward},
    tensor::assert_image_patchable,
};

#[derive(Debug)]
struct MaxViTAttentionBlock {
    attention: Attention,
    feed_forward: FeedForward,
    window_size: i64,
    grid: bool,
}

impl MaxViTAttentionBlock {
    fn new(
        vs: &nn::Path,
        dim: i64,
        dim_head: i64,
        window_size: i64,
        dropout: f64,
        grid: bool,
    ) -> Self {
        assert_eq!(dim % dim_head, 0, "dim must be divisible by dim head");
        let heads = dim / dim_head;
        let attention = Attention::new(&(vs / "attention"), dim, heads, dim_head, dropout);
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), dim, dim * 4, dropout);

        Self {
            attention,
            feed_forward,
            window_size,
            grid,
        }
    }

    fn block_tokens(&self, xs: &Tensor) -> (Tensor, i64, i64, i64, i64) {
        let size = xs.size();
        let batch = size[0];
        let dim = size[1];
        let height = size[2];
        let width = size[3];
        assert_image_patchable(height, width, self.window_size, self.window_size);
        let grid_h = height / self.window_size;
        let grid_w = width / self.window_size;
        let tokens = if self.grid {
            xs.view([batch, dim, self.window_size, grid_h, self.window_size, grid_w])
                .permute([0, 3, 5, 2, 4, 1])
                .contiguous()
        } else {
            xs.view([batch, dim, grid_h, self.window_size, grid_w, self.window_size])
                .permute([0, 2, 4, 3, 5, 1])
                .contiguous()
        };

        (
            tokens.view([batch * grid_h * grid_w, self.window_size * self.window_size, dim]),
            batch,
            height,
            width,
            dim,
        )
    }

    fn tokens_to_blocks(&self, tokens: &Tensor, batch: i64, height: i64, width: i64, dim: i64) -> Tensor {
        let grid_h = height / self.window_size;
        let grid_w = width / self.window_size;
        let tokens = tokens.view([
            batch,
            grid_h,
            grid_w,
            self.window_size,
            self.window_size,
            dim,
        ]);

        if self.grid {
            tokens
                .permute([0, 5, 3, 1, 4, 2])
                .contiguous()
                .view([batch, dim, height, width])
        } else {
            tokens
                .permute([0, 5, 1, 3, 2, 4])
                .contiguous()
                .view([batch, dim, height, width])
        }
    }
}

impl nn::ModuleT for MaxViTAttentionBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let (tokens, batch, height, width, dim) = self.block_tokens(xs);
        let tokens = &tokens + self.attention.forward_t(&tokens, train);
        let ff_out = self.feed_forward.forward_t(&tokens, train);
        let tokens = tokens + ff_out;

        self.tokens_to_blocks(&tokens, batch, height, width, dim)
    }
}

#[derive(Debug)]
struct MaxViTLayer {
    mbconv: MbConv,
    block_attention: MaxViTAttentionBlock,
    grid_attention: MaxViTAttentionBlock,
}

impl MaxViTLayer {
    fn new(
        vs: &nn::Path,
        dim_in: i64,
        dim_out: i64,
        downsample: bool,
        config: &MaxViTConfig,
    ) -> Self {
        let stride = if downsample { 2 } else { 1 };
        let mbconv = MbConv::new(
            &(vs / "mbconv"),
            dim_in,
            dim_out,
            stride,
            config.mbconv_expansion_rate,
            config.mbconv_shrinkage_rate,
            config.dropout,
        );
        let block_attention = MaxViTAttentionBlock::new(
            &(vs / "block_attention"),
            dim_out,
            config.dim_head,
            config.window_size,
            config.dropout,
            false,
        );
        let grid_attention = MaxViTAttentionBlock::new(
            &(vs / "grid_attention"),
            dim_out,
            config.dim_head,
            config.window_size,
            config.dropout,
            true,
        );

        Self {
            mbconv,
            block_attention,
            grid_attention,
        }
    }
}

impl nn::ModuleT for MaxViTLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = self.mbconv.forward_t(xs, train);
        let xs = self.block_attention.forward_t(&xs, train);

        self.grid_attention.forward_t(&xs, train)
    }
}

#[derive(Debug)]
pub struct MaxViT {
    conv_stem1: nn::Conv2D,
    conv_stem2: nn::Conv2D,
    layers: Vec<MaxViTLayer>,
    head_norm: nn::LayerNorm,
    head: nn::Linear,
}

impl MaxViT {
    pub fn new(vs: &nn::Path, config: MaxViTConfig) -> Self {
        assert!(!config.depth.is_empty(), "depth must not be empty");
        let dim_conv_stem = config.dim_conv_stem.unwrap_or(config.dim);
        let conv_stem1 = nn::conv2d(
            vs / "conv_stem1",
            config.channels,
            dim_conv_stem,
            3,
            nn::ConvConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
        );
        let conv_stem2 = nn::conv2d(
            vs / "conv_stem2",
            dim_conv_stem,
            dim_conv_stem,
            3,
            nn::ConvConfig {
                padding: 1,
                ..Default::default()
            },
        );
        let mut dims = vec![dim_conv_stem];
        dims.extend((0..config.depth.len()).map(|index| config.dim * 2_i64.pow(index as u32)));
        let mut layers = Vec::new();

        for stage_index in 0..config.depth.len() {
            let dim_in = dims[stage_index];
            let dim_out = dims[stage_index + 1];

            for layer_index in 0..config.depth[stage_index] {
                let name = format!("stage_{stage_index}_layer_{layer_index}");
                let current_dim_in = if layer_index == 0 { dim_in } else { dim_out };
                layers.push(MaxViTLayer::new(
                    &(vs / name.as_str()),
                    current_dim_in,
                    dim_out,
                    layer_index == 0,
                    &config,
                ));
            }
        }

        let final_dim = *dims.last().expect("final dim");
        let head_norm = nn::layer_norm(vs / "head_norm", vec![final_dim], Default::default());
        let head = nn::linear(
            vs / "head",
            final_dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            conv_stem1,
            conv_stem2,
            layers,
            head_norm,
            head,
        }
    }
}

impl nn::ModuleT for MaxViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.apply(&self.conv_stem1).apply(&self.conv_stem2);

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs.mean_dim(&[2_i64, 3][..], false, Kind::Float)
            .apply(&self.head_norm)
            .apply(&self.head)
    }
}
