use tch::{IndexOp, Kind, Tensor, nn, nn::ModuleT};

use crate::{
    config::XCiTConfig,
    layers::FeedForward,
    models::vit::PatchEmbedding,
    norm::l2_normalize,
    tensor::{merge_heads, num_patches, patch_dim, repeat_token, split_heads},
};

fn layer_scale_init(depth: usize) -> f64 {
    if depth <= 18 {
        0.1
    } else if depth <= 24 {
        1e-5
    } else {
        1e-6
    }
}

#[derive(Debug)]
struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    fn new(vs: &nn::Path, dim: i64, depth: usize) -> Self {
        let scale = vs.var(
            "scale",
            &[1, 1, dim],
            nn::Init::Const(layer_scale_init(depth)),
        );

        Self { scale }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs * &self.scale
    }
}

#[derive(Debug)]
struct XCAttention {
    norm: nn::LayerNorm,
    to_qkv: nn::Linear,
    temperature: Tensor,
    to_out: nn::Linear,
    heads: i64,
    dim_head: i64,
    dropout: f64,
}

impl XCAttention {
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
        let temperature = vs.var("temperature", &[heads, 1, 1], nn::Init::Const(1.0));
        let to_out = nn::linear(vs / "to_out", inner_dim, dim, Default::default());

        Self {
            norm,
            to_qkv,
            temperature,
            to_out,
            heads,
            dim_head,
            dropout,
        }
    }
}

impl nn::ModuleT for XCAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        assert_eq!(
            size.len(),
            4,
            "expected feature map [batch, height, width, dim]"
        );
        let batch = size[0];
        let height = size[1];
        let width = size[2];
        let dim = size[3];
        let tokens = height * width;
        let xs = xs
            .view([batch, tokens, dim])
            .apply(&self.norm)
            .apply(&self.to_qkv)
            .chunk(3, -1);
        let q = xs[0]
            .view([batch, tokens, self.heads, self.dim_head])
            .permute([0, 2, 3, 1]);
        let k = xs[1]
            .view([batch, tokens, self.heads, self.dim_head])
            .permute([0, 2, 3, 1]);
        let v = xs[2]
            .view([batch, tokens, self.heads, self.dim_head])
            .permute([0, 2, 3, 1]);
        let q = l2_normalize(&q, -1, 1e-12);
        let k = l2_normalize(&k, -1, 1e-12);
        let attn = (q.matmul(&k.transpose(-1, -2)) * self.temperature.exp().unsqueeze(0))
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train);
        let out = attn
            .matmul(&v)
            .permute([0, 3, 1, 2])
            .contiguous()
            .view([batch, tokens, self.heads * self.dim_head])
            .apply(&self.to_out)
            .dropout(self.dropout, train);

        out.view([batch, height, width, dim])
    }
}

#[derive(Debug)]
struct LocalPatchInteraction {
    norm: nn::LayerNorm,
    depthwise1: nn::Conv2D,
    bn: nn::BatchNorm,
    depthwise2: nn::Conv2D,
}

impl LocalPatchInteraction {
    fn new(vs: &nn::Path, dim: i64, kernel_size: i64) -> Self {
        assert!(
            kernel_size > 0 && kernel_size % 2 == 1,
            "local patch kernel size must be positive and odd"
        );
        let padding = kernel_size / 2;
        let norm = nn::layer_norm(vs / "norm", vec![dim], Default::default());
        let conv_config = nn::ConvConfig {
            padding,
            groups: dim,
            ..Default::default()
        };
        let depthwise1 = nn::conv2d(vs / "depthwise1", dim, dim, kernel_size, conv_config);
        let bn = nn::batch_norm2d(vs / "bn", dim, Default::default());
        let depthwise2 = nn::conv2d(vs / "depthwise2", dim, dim, kernel_size, conv_config);

        Self {
            norm,
            depthwise1,
            bn,
            depthwise2,
        }
    }
}

impl nn::ModuleT for LocalPatchInteraction {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        assert_eq!(
            size.len(),
            4,
            "expected feature map [batch, height, width, dim]"
        );
        let batch = size[0];
        let height = size[1];
        let width = size[2];
        let dim = size[3];

        xs.view([batch, height * width, dim])
            .apply(&self.norm)
            .view([batch, height, width, dim])
            .permute([0, 3, 1, 2])
            .apply(&self.depthwise1)
            .apply_t(&self.bn, train)
            .gelu("none")
            .apply(&self.depthwise2)
            .permute([0, 2, 3, 1])
    }
}

#[derive(Debug)]
struct XCiTLayer {
    cross_covariance_attention: XCAttention,
    local_patch_interaction: LocalPatchInteraction,
    feed_forward: FeedForward,
    attention_scale: LayerScale,
    interaction_scale: LayerScale,
    feed_forward_scale: LayerScale,
}

impl XCiTLayer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        local_patch_kernel_size: i64,
        dropout: f64,
        depth: usize,
    ) -> Self {
        let cross_covariance_attention = XCAttention::new(
            &(vs / "cross_covariance_attention"),
            dim,
            heads,
            dim_head,
            dropout,
        );
        let local_patch_interaction = LocalPatchInteraction::new(
            &(vs / "local_patch_interaction"),
            dim,
            local_patch_kernel_size,
        );
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), dim, mlp_dim, dropout);
        let attention_scale = LayerScale::new(&(vs / "attention_scale"), dim, depth);
        let interaction_scale = LayerScale::new(&(vs / "interaction_scale"), dim, depth);
        let feed_forward_scale = LayerScale::new(&(vs / "feed_forward_scale"), dim, depth);

        Self {
            cross_covariance_attention,
            local_patch_interaction,
            feed_forward,
            attention_scale,
            interaction_scale,
            feed_forward_scale,
        }
    }
}

impl nn::ModuleT for XCiTLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let attn = self
            .attention_scale
            .forward(&self.cross_covariance_attention.forward_t(xs, train));
        let xs = xs + attn;
        let interaction = self
            .interaction_scale
            .forward(&self.local_patch_interaction.forward_t(&xs, train));
        let xs = xs + interaction;
        let size = xs.size();
        let batch = size[0];
        let height = size[1];
        let width = size[2];
        let dim = size[3];
        let tokens = xs.view([batch, height * width, dim]);
        let ff = self
            .feed_forward_scale
            .forward(&self.feed_forward.forward_t(&tokens, train))
            .view([batch, height, width, dim]);

        xs + ff
    }
}

#[derive(Debug)]
struct XCATransformer {
    layers: Vec<XCiTLayer>,
    layer_dropout: f64,
}

impl XCATransformer {
    fn new(vs: &nn::Path, config: &XCiTConfig) -> Self {
        assert!(
            config.layer_dropout >= 0.0 && config.layer_dropout < 1.0,
            "layer dropout must be in [0, 1)"
        );
        let layers = (0..config.depth)
            .map(|index| {
                let name = format!("layer_{index}");
                XCiTLayer::new(
                    &(vs / name.as_str()),
                    config.dim,
                    config.heads,
                    config.dim_head,
                    config.mlp_dim,
                    config.local_patch_kernel_size,
                    config.dropout,
                    index + 1,
                )
            })
            .collect();

        Self {
            layers,
            layer_dropout: config.layer_dropout,
        }
    }
}

impl nn::ModuleT for XCATransformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            if train && self.layer_dropout > 0.0 {
                let drop = Tensor::rand([], (xs.kind(), xs.device())).double_value(&[])
                    < self.layer_dropout;
                if drop {
                    continue;
                }
            }

            xs = layer.forward_t(&xs, train);
        }

        xs
    }
}

#[derive(Debug)]
struct ClassAttention {
    norm: nn::LayerNorm,
    to_q: nn::Linear,
    to_kv: nn::Linear,
    to_out: nn::Linear,
    heads: i64,
    scale: f64,
    dropout: f64,
}

impl ClassAttention {
    fn new(vs: &nn::Path, dim: i64, heads: i64, dim_head: i64, dropout: f64) -> Self {
        assert!(heads > 0, "heads must be positive");
        assert!(dim_head > 0, "dim head must be positive");
        let inner_dim = heads * dim_head;
        let norm = nn::layer_norm(vs / "norm", vec![dim], Default::default());
        let linear_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        let to_q = nn::linear(vs / "to_q", dim, inner_dim, linear_config);
        let to_kv = nn::linear(vs / "to_kv", dim, inner_dim * 2, linear_config);
        let to_out = nn::linear(vs / "to_out", inner_dim, dim, Default::default());

        Self {
            norm,
            to_q,
            to_kv,
            to_out,
            heads,
            scale: (dim_head as f64).powf(-0.5),
            dropout,
        }
    }

    fn forward_t_with_context(&self, xs: &Tensor, context: &Tensor, train: bool) -> Tensor {
        let xs = xs.apply(&self.norm);
        let context = Tensor::cat(&[&xs, context], 1);
        let q = split_heads(&xs.apply(&self.to_q), self.heads);
        let kv = context.apply(&self.to_kv).chunk(2, -1);
        let k = split_heads(&kv[0], self.heads);
        let v = split_heads(&kv[1], self.heads);
        let attn = (q.matmul(&k.transpose(-1, -2)) * self.scale)
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train);

        merge_heads(&attn.matmul(&v))
            .apply(&self.to_out)
            .dropout(self.dropout, train)
    }
}

#[derive(Debug)]
struct ClassTransformerLayer {
    attention: ClassAttention,
    feed_forward: FeedForward,
    attention_scale: LayerScale,
    feed_forward_scale: LayerScale,
}

impl ClassTransformerLayer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        dropout: f64,
        depth: usize,
    ) -> Self {
        let attention = ClassAttention::new(&(vs / "attention"), dim, heads, dim_head, dropout);
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), dim, mlp_dim, dropout);
        let attention_scale = LayerScale::new(&(vs / "attention_scale"), dim, depth);
        let feed_forward_scale = LayerScale::new(&(vs / "feed_forward_scale"), dim, depth);

        Self {
            attention,
            feed_forward,
            attention_scale,
            feed_forward_scale,
        }
    }

    fn forward_t_with_context(&self, xs: &Tensor, context: &Tensor, train: bool) -> Tensor {
        let attn = self
            .attention_scale
            .forward(&self.attention.forward_t_with_context(xs, context, train));
        let xs = xs + attn;
        let ff = self
            .feed_forward_scale
            .forward(&self.feed_forward.forward_t(&xs, train));

        xs + ff
    }
}

#[derive(Debug)]
struct ClassTransformer {
    layers: Vec<ClassTransformerLayer>,
    layer_dropout: f64,
}

impl ClassTransformer {
    fn new(vs: &nn::Path, config: &XCiTConfig) -> Self {
        let layers = (0..config.cls_depth)
            .map(|index| {
                let name = format!("layer_{index}");
                ClassTransformerLayer::new(
                    &(vs / name.as_str()),
                    config.dim,
                    config.heads,
                    config.dim_head,
                    config.mlp_dim,
                    config.dropout,
                    index + 1,
                )
            })
            .collect();

        Self {
            layers,
            layer_dropout: config.layer_dropout,
        }
    }

    fn forward_t_with_context(&self, xs: &Tensor, context: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            if train && self.layer_dropout > 0.0 {
                let drop = Tensor::rand([], (xs.kind(), xs.device())).double_value(&[])
                    < self.layer_dropout;
                if drop {
                    continue;
                }
            }

            xs = layer.forward_t_with_context(&xs, context, train);
        }

        xs
    }
}

#[derive(Debug)]
pub struct XCiT {
    patch_embedding: PatchEmbedding,
    pos_embedding: Tensor,
    cls_token: Tensor,
    xcit_transformer: XCATransformer,
    final_norm: nn::LayerNorm,
    cls_transformer: ClassTransformer,
    mlp_norm: nn::LayerNorm,
    mlp_head: nn::Linear,
    emb_dropout: f64,
    patch_height: i64,
    patch_width: i64,
}

impl XCiT {
    pub fn new(vs: &nn::Path, config: XCiTConfig) -> Self {
        assert!(
            config.depth > 0 || config.cls_depth > 0,
            "XCiT needs at least one transformer layer"
        );
        let patch_height = config.patch_size.height;
        let patch_width = config.patch_size.width;
        let num_patches = num_patches(
            config.image_size.height,
            config.image_size.width,
            patch_height,
            patch_width,
        );
        let patch_dim = patch_dim(config.channels, patch_height, patch_width);
        let patch_embedding = PatchEmbedding::new(
            &(vs / "patch_embedding"),
            patch_dim,
            config.dim,
            patch_height,
            patch_width,
        );
        let pos_embedding = vs.var(
            "pos_embedding",
            &[num_patches, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let cls_token = vs.var(
            "cls_token",
            &[config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let xcit_transformer = XCATransformer::new(&(vs / "xcit_transformer"), &config);
        let final_norm = nn::layer_norm(vs / "final_norm", vec![config.dim], Default::default());
        let cls_transformer = ClassTransformer::new(&(vs / "cls_transformer"), &config);
        let mlp_norm = nn::layer_norm(vs / "mlp_norm", vec![config.dim], Default::default());
        let mlp_head = nn::linear(
            vs / "mlp_head",
            config.dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            patch_embedding,
            pos_embedding,
            cls_token,
            xcit_transformer,
            final_norm,
            cls_transformer,
            mlp_norm,
            mlp_head,
            emb_dropout: config.emb_dropout,
            patch_height,
            patch_width,
        }
    }
}

impl nn::ModuleT for XCiT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let batch = xs.size()[0];
        let height = xs.size()[2] / self.patch_height;
        let width = xs.size()[3] / self.patch_width;
        let mut tokens = xs.apply(&self.patch_embedding);
        let tokens_len = tokens.size()[1];
        let pos_embedding = self.pos_embedding.i(0..tokens_len).unsqueeze(0);
        tokens = (tokens + pos_embedding).dropout(self.emb_dropout, train);
        let tokens = tokens.view([batch, height, width, -1]);
        let tokens = self
            .xcit_transformer
            .forward_t(&tokens, train)
            .view([batch, height * width, -1])
            .apply(&self.final_norm);
        let cls_token = repeat_token(&self.cls_token, batch);
        let cls_token = self
            .cls_transformer
            .forward_t_with_context(&cls_token, &tokens, train);

        cls_token
            .i((.., 0))
            .apply(&self.mlp_norm)
            .apply(&self.mlp_head)
    }
}
