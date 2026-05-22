use tch::{IndexOp, Tensor, nn, nn::ModuleT};

use crate::{
    config::CaiTConfig,
    layers::FeedForward,
    models::vit::PatchEmbedding,
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
struct CaiTAttention {
    norm: nn::LayerNorm,
    to_q: nn::Linear,
    to_kv: nn::Linear,
    mix_heads_pre_attn: Tensor,
    mix_heads_post_attn: Tensor,
    to_out: nn::Linear,
    heads: i64,
    scale: f64,
    dropout: f64,
}

impl CaiTAttention {
    fn new(vs: &nn::Path, dim: i64, heads: i64, dim_head: i64, dropout: f64) -> Self {
        let inner_dim = heads * dim_head;
        let norm = nn::layer_norm(vs / "norm", vec![dim], Default::default());
        let linear_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        let to_q = nn::linear(vs / "to_q", dim, inner_dim, linear_config);
        let to_kv = nn::linear(vs / "to_kv", dim, inner_dim * 2, linear_config);
        let mix_heads_pre_attn = vs.var(
            "mix_heads_pre_attn",
            &[heads, heads],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let mix_heads_post_attn = vs.var(
            "mix_heads_post_attn",
            &[heads, heads],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let to_out = nn::linear(vs / "to_out", inner_dim, dim, Default::default());

        Self {
            norm,
            to_q,
            to_kv,
            mix_heads_pre_attn,
            mix_heads_post_attn,
            to_out,
            heads,
            scale: (dim_head as f64).powf(-0.5),
            dropout,
        }
    }

    fn forward_t_with_context(&self, xs: &Tensor, context: Option<&Tensor>, train: bool) -> Tensor {
        let xs = xs.apply(&self.norm);
        let kv_input = match context {
            Some(context) => Tensor::cat(&[&xs, context], 1),
            None => xs.shallow_clone(),
        };
        let q = split_heads(&xs.apply(&self.to_q), self.heads);
        let kv = kv_input.apply(&self.to_kv).chunk(2, -1);
        let k = split_heads(&kv[0], self.heads);
        let v = split_heads(&kv[1], self.heads);
        let dots = q.matmul(&k.transpose(-1, -2)) * self.scale;
        let dots = dots
            .permute([0, 2, 3, 1])
            .matmul(&self.mix_heads_pre_attn)
            .permute([0, 3, 1, 2]);
        let attn = dots.softmax(-1, dots.kind()).dropout(self.dropout, train);
        let attn = attn
            .permute([0, 2, 3, 1])
            .matmul(&self.mix_heads_post_attn)
            .permute([0, 3, 1, 2]);

        merge_heads(&attn.matmul(&v))
            .apply(&self.to_out)
            .dropout(self.dropout, train)
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
struct CaiTLayer {
    attention: CaiTAttention,
    feed_forward: FeedForward,
    attention_scale: LayerScale,
    feed_forward_scale: LayerScale,
}

impl CaiTLayer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        dropout: f64,
        depth: usize,
    ) -> Self {
        let attention = CaiTAttention::new(&(vs / "attention"), dim, heads, dim_head, dropout);
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

    fn forward_t_with_context(&self, xs: &Tensor, context: Option<&Tensor>, train: bool) -> Tensor {
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
struct CaiTTransformer {
    layers: Vec<CaiTLayer>,
    layer_dropout: f64,
}

impl CaiTTransformer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        depth: usize,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        dropout: f64,
        layer_dropout: f64,
    ) -> Self {
        assert!(
            layer_dropout >= 0.0 && layer_dropout < 1.0,
            "layer dropout must be in [0, 1)"
        );
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                CaiTLayer::new(
                    &(vs / name.as_str()),
                    dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    dropout,
                    index + 1,
                )
            })
            .collect();

        Self {
            layers,
            layer_dropout,
        }
    }

    fn forward_t_with_context(&self, xs: &Tensor, context: Option<&Tensor>, train: bool) -> Tensor {
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

impl nn::ModuleT for CaiTTransformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward_t_with_context(xs, None, train)
    }
}

#[derive(Debug)]
pub struct CaiT {
    patch_embedding: PatchEmbedding,
    pos_embedding: Tensor,
    cls_token: Tensor,
    patch_transformer: CaiTTransformer,
    cls_transformer: CaiTTransformer,
    mlp_norm: nn::LayerNorm,
    mlp_head: nn::Linear,
    emb_dropout: f64,
}

impl CaiT {
    pub fn new(vs: &nn::Path, config: CaiTConfig) -> Self {
        assert_eq!(
            config.image_size.height, config.image_size.width,
            "CaiT expects square image size"
        );
        assert_eq!(
            config.patch_size.height, config.patch_size.width,
            "CaiT expects square patch size"
        );
        assert!(
            config.depth > 0 || config.cls_depth > 0,
            "CaiT needs at least one transformer layer"
        );
        let patch_height = config.patch_size.height;
        let patch_width = config.patch_size.width;
        let patch_dim = patch_dim(config.channels, patch_height, patch_width);
        let num_patches = num_patches(
            config.image_size.height,
            config.image_size.width,
            patch_height,
            patch_width,
        );
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
            &[1, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let patch_transformer = CaiTTransformer::new(
            &(vs / "patch_transformer"),
            config.dim,
            config.depth,
            config.heads,
            config.dim_head,
            config.mlp_dim,
            config.dropout,
            config.layer_dropout,
        );
        let cls_transformer = CaiTTransformer::new(
            &(vs / "cls_transformer"),
            config.dim,
            config.cls_depth,
            config.heads,
            config.dim_head,
            config.mlp_dim,
            config.dropout,
            config.layer_dropout,
        );
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
            patch_transformer,
            cls_transformer,
            mlp_norm,
            mlp_head,
            emb_dropout: config.emb_dropout,
        }
    }
}

impl nn::ModuleT for CaiT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let batch = xs.size()[0];
        let mut xs = xs.apply(&self.patch_embedding);
        let seq_len = xs.size()[1];
        let pos_embedding = self.pos_embedding.i(0..seq_len).unsqueeze(0);
        xs = (xs + pos_embedding).dropout(self.emb_dropout, train);
        xs = self.patch_transformer.forward_t(&xs, train);

        let cls_tokens = repeat_token(&self.cls_token, batch);
        let cls_tokens = self
            .cls_transformer
            .forward_t_with_context(&cls_tokens, Some(&xs), train);

        cls_tokens
            .i((.., 0))
            .apply(&self.mlp_norm)
            .apply(&self.mlp_head)
    }
}
