use tch::{Kind, Tensor, nn, nn::ModuleT};

use crate::config::LeViTConfig;

#[derive(Debug)]
struct ConvBn {
    conv: nn::Conv2D,
    bn: nn::BatchNorm,
}

impl ConvBn {
    fn new(vs: &nn::Path, dim_in: i64, dim_out: i64, kernel_size: i64, stride: i64) -> Self {
        assert!(kernel_size > 0, "kernel size must be positive");
        assert!(stride > 0, "stride must be positive");
        let conv = nn::conv2d(
            vs / "conv",
            dim_in,
            dim_out,
            kernel_size,
            nn::ConvConfig {
                stride,
                padding: kernel_size / 2,
                bias: false,
                ..Default::default()
            },
        );
        let bn = nn::batch_norm2d(vs / "bn", dim_out, Default::default());

        Self { conv, bn }
    }
}

impl nn::ModuleT for ConvBn {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.conv).apply_t(&self.bn, train)
    }
}

#[derive(Debug)]
struct ConvEmbedding {
    layers: Vec<nn::Conv2D>,
}

impl ConvEmbedding {
    fn new(vs: &nn::Path, channels: i64, dim: i64) -> Self {
        let layer_dims = [(channels, 32), (32, 64), (64, 128), (128, dim)];
        let layers = layer_dims
            .iter()
            .enumerate()
            .map(|(index, (dim_in, dim_out))| {
                let name = format!("layer_{index}");
                nn::conv2d(
                    vs / name.as_str(),
                    *dim_in,
                    *dim_out,
                    3,
                    nn::ConvConfig {
                        stride: 2,
                        padding: 1,
                        ..Default::default()
                    },
                )
            })
            .collect();

        Self { layers }
    }
}

impl nn::ModuleT for ConvEmbedding {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            xs = xs.apply(layer);
        }

        xs
    }
}

#[derive(Debug)]
struct FeedForward2D {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    dropout: f64,
}

impl FeedForward2D {
    fn new(vs: &nn::Path, dim: i64, mult: i64, dropout: f64) -> Self {
        assert!(mult > 0, "mlp multiplier must be positive");
        let hidden_dim = dim * mult;
        let conv1 = nn::conv2d(vs / "conv1", dim, hidden_dim, 1, Default::default());
        let conv2 = nn::conv2d(vs / "conv2", hidden_dim, dim, 1, Default::default());

        Self {
            conv1,
            conv2,
            dropout,
        }
    }
}

impl nn::ModuleT for FeedForward2D {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.conv1)
            .hardswish()
            .dropout(self.dropout, train)
            .apply(&self.conv2)
            .dropout(self.dropout, train)
    }
}

#[derive(Debug)]
struct ConvAttention2D {
    to_q: ConvBn,
    to_k: ConvBn,
    to_v: ConvBn,
    to_out: nn::Conv2D,
    out_bn: nn::BatchNorm,
    pos_bias: Tensor,
    pos_indices: Tensor,
    heads: i64,
    dim_key: i64,
    dim_value: i64,
    scale: f64,
    dropout: f64,
}

impl ConvAttention2D {
    fn new(
        vs: &nn::Path,
        dim: i64,
        fmap_size: i64,
        heads: i64,
        dim_key: i64,
        dim_value: i64,
        dropout: f64,
        dim_out: i64,
        downsample: bool,
    ) -> Self {
        assert!(fmap_size > 0, "feature map size must be positive");
        assert!(heads > 0, "heads must be positive");
        assert!(dim_key > 0, "key dimension must be positive");
        assert!(dim_value > 0, "value dimension must be positive");
        let q_stride = if downsample { 2 } else { 1 };
        let inner_dim_key = heads * dim_key;
        let inner_dim_value = heads * dim_value;
        let to_q = ConvBn::new(&(vs / "to_q"), dim, inner_dim_key, 1, q_stride);
        let to_k = ConvBn::new(&(vs / "to_k"), dim, inner_dim_key, 1, 1);
        let to_v = ConvBn::new(&(vs / "to_v"), dim, inner_dim_value, 1, 1);
        let to_out = nn::conv2d(
            vs / "to_out",
            inner_dim_value,
            dim_out,
            1,
            Default::default(),
        );
        let out_bn = nn::batch_norm2d(
            vs / "out_bn",
            dim_out,
            nn::BatchNormConfig {
                ws_init: nn::Init::Const(0.0),
                ..Default::default()
            },
        );
        let pos_bias = vs.var(
            "pos_bias",
            &[fmap_size * fmap_size, heads],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let pos_indices = relative_position_indices(fmap_size, q_stride, vs.device());

        Self {
            to_q,
            to_k,
            to_v,
            to_out,
            out_bn,
            pos_bias,
            pos_indices,
            heads,
            dim_key,
            dim_value,
            scale: (dim_key as f64).powf(-0.5),
            dropout,
        }
    }
}

impl nn::ModuleT for ConvAttention2D {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let batch = xs.size()[0];
        let q = self.to_q.forward_t(xs, train);
        let q_height = q.size()[2];
        let q_width = q.size()[3];
        let k = self.to_k.forward_t(xs, train);
        let v = self.to_v.forward_t(xs, train);
        let q_tokens = q_height * q_width;
        let k_tokens = k.size()[2] * k.size()[3];
        let q = q
            .view([batch, self.heads, self.dim_key, q_tokens])
            .transpose(2, 3);
        let k = k
            .view([batch, self.heads, self.dim_key, k_tokens])
            .transpose(2, 3);
        let v = v
            .view([batch, self.heads, self.dim_value, k_tokens])
            .transpose(2, 3);
        let bias = self
            .pos_bias
            .index_select(0, &self.pos_indices.view([-1]))
            .view([q_tokens, k_tokens, self.heads])
            .permute([2, 0, 1])
            .unsqueeze(0);
        let attn = (q.matmul(&k.transpose(-1, -2)) * self.scale + bias / self.scale)
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train);
        let out = attn.matmul(&v).transpose(2, 3).contiguous().view([
            batch,
            self.heads * self.dim_value,
            q_height,
            q_width,
        ]);
        out.gelu("none")
            .apply(&self.to_out)
            .apply_t(&self.out_bn, train)
            .dropout(self.dropout, train)
    }
}

#[derive(Debug)]
struct LeViTLayer {
    attention: ConvAttention2D,
    feed_forward: FeedForward2D,
    attn_residual: bool,
}

impl LeViTLayer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        fmap_size: i64,
        heads: i64,
        dim_key: i64,
        dim_value: i64,
        mlp_mult: i64,
        dropout: f64,
        dim_out: i64,
        downsample: bool,
    ) -> Self {
        let attention = ConvAttention2D::new(
            &(vs / "attention"),
            dim,
            fmap_size,
            heads,
            dim_key,
            dim_value,
            dropout,
            dim_out,
            downsample,
        );
        let feed_forward = FeedForward2D::new(&(vs / "feed_forward"), dim_out, mlp_mult, dropout);
        let attn_residual = !downsample && dim == dim_out;

        Self {
            attention,
            feed_forward,
            attn_residual,
        }
    }
}

impl nn::ModuleT for LeViTLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let attn_res = if self.attn_residual {
            xs.shallow_clone()
        } else {
            Tensor::zeros([], (xs.kind(), xs.device()))
        };
        let xs = self.attention.forward_t(xs, train) + attn_res;
        let ff_out = self.feed_forward.forward_t(&xs, train);

        xs + ff_out
    }
}

#[derive(Debug)]
struct LeViTTransformer {
    layers: Vec<LeViTLayer>,
}

impl LeViTTransformer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        fmap_size: i64,
        depth: usize,
        heads: i64,
        dim_key: i64,
        dim_value: i64,
        mlp_mult: i64,
        dropout: f64,
        dim_out: i64,
        downsample: bool,
    ) -> Self {
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                LeViTLayer::new(
                    &(vs / name.as_str()),
                    dim,
                    fmap_size,
                    heads,
                    dim_key,
                    dim_value,
                    mlp_mult,
                    dropout,
                    dim_out,
                    downsample,
                )
            })
            .collect();

        Self { layers }
    }
}

impl nn::ModuleT for LeViTTransformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs
    }
}

#[derive(Debug)]
pub struct LeViT {
    conv_embedding: ConvEmbedding,
    backbone: Vec<LeViTTransformer>,
    mlp_head: nn::Linear,
    distill_head: Option<nn::Linear>,
}

impl LeViT {
    pub fn new(vs: &nn::Path, config: LeViTConfig) -> Self {
        assert!(config.image_size > 0, "image size must be positive");
        assert_eq!(
            config.image_size % 16,
            0,
            "image size must be divisible by the convolutional embedding stride"
        );
        assert!(config.stages > 0, "LeViT requires at least one stage");
        assert_eq!(
            config.dims.len(),
            config.stages,
            "LeViT dims must match stage count"
        );
        assert_eq!(
            config.depths.len(),
            config.stages,
            "LeViT depths must match stage count"
        );
        assert_eq!(
            config.heads.len(),
            config.stages,
            "LeViT heads must match stage count"
        );
        let conv_embedding =
            ConvEmbedding::new(&(vs / "conv_embedding"), config.channels, config.dims[0]);
        let mut fmap_size = config.image_size / 16;
        let mut backbone = Vec::with_capacity(config.stages * 2 - 1);

        for index in 0..config.stages {
            let stage_name = format!("stage_{index}");
            let is_last = index + 1 == config.stages;
            backbone.push(LeViTTransformer::new(
                &(vs / stage_name.as_str()),
                config.dims[index],
                fmap_size,
                config.depths[index],
                config.heads[index],
                config.dim_key,
                config.dim_value,
                config.mlp_mult,
                config.dropout,
                config.dims[index],
                false,
            ));

            if !is_last {
                let downsample_name = format!("downsample_{index}");
                backbone.push(LeViTTransformer::new(
                    &(vs / downsample_name.as_str()),
                    config.dims[index],
                    fmap_size,
                    1,
                    config.heads[index] * 2,
                    config.dim_key,
                    config.dim_value,
                    config.mlp_mult,
                    config.dropout,
                    config.dims[index + 1],
                    true,
                ));
                fmap_size = ceil_div(fmap_size, 2);
            }
        }

        let final_dim = config.dims[config.stages - 1];
        let distill_head = config
            .num_distill_classes
            .map(|classes| nn::linear(vs / "distill_head", final_dim, classes, Default::default()));
        let mlp_head = nn::linear(
            vs / "mlp_head",
            final_dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            conv_embedding,
            backbone,
            mlp_head,
            distill_head,
        }
    }

    pub fn forward_t_with_distill(&self, xs: &Tensor, train: bool) -> (Tensor, Option<Tensor>) {
        let features = self.forward_features_t(xs, train);
        let logits = features.apply(&self.mlp_head);
        let distill = self.distill_head.as_ref().map(|head| features.apply(head));

        (logits, distill)
    }

    fn forward_features_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = self.conv_embedding.forward_t(xs, train);

        for layer in &self.backbone {
            xs = layer.forward_t(&xs, train);
        }

        xs.adaptive_avg_pool2d([1, 1]).flat_view()
    }
}

impl nn::ModuleT for LeViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward_t_with_distill(xs, train).0
    }
}

fn relative_position_indices(fmap_size: i64, q_stride: i64, device: tch::Device) -> Tensor {
    let q_coords = stepped_grid_coords(fmap_size, q_stride);
    let k_coords = stepped_grid_coords(fmap_size, 1);
    let mut indices = Vec::with_capacity(q_coords.len() * k_coords.len());

    for (qx, qy) in &q_coords {
        for (kx, ky) in &k_coords {
            indices.push((qx - kx).abs() * fmap_size + (qy - ky).abs());
        }
    }

    Tensor::from_slice(&indices)
        .to_device(device)
        .view([q_coords.len() as i64, k_coords.len() as i64])
}

fn stepped_grid_coords(size: i64, step: i64) -> Vec<(i64, i64)> {
    let mut coords = Vec::new();
    let mut x = 0;

    while x < size {
        let mut y = 0;

        while y < size {
            coords.push((x, y));
            y += step;
        }

        x += step;
    }

    coords
}

fn ceil_div(value: i64, divisor: i64) -> i64 {
    (value + divisor - 1) / divisor
}
