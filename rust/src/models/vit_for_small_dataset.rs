use tch::{IndexOp, Tensor, nn};

use crate::{
    config::{Pool, ViTConfig},
    layers::FeedForward,
    tensor::{merge_heads, num_patches, patch_dim, patchify_2d, repeat_token, split_heads},
};

#[derive(Debug)]
struct ShiftedPatchEmbedding {
    norm: nn::LayerNorm,
    linear: nn::Linear,
    patch_height: i64,
    patch_width: i64,
}

impl ShiftedPatchEmbedding {
    fn new(
        vs: &nn::Path,
        channels: i64,
        dim: i64,
        patch_height: i64,
        patch_width: i64,
    ) -> Self {
        let patch_dim = patch_dim(channels * 5, patch_height, patch_width);
        let norm = nn::layer_norm(vs / "norm", vec![patch_dim], Default::default());
        let linear = nn::linear(vs / "linear", patch_dim, dim, Default::default());

        Self {
            norm,
            linear,
            patch_height,
            patch_width,
        }
    }
}

impl nn::Module for ShiftedPatchEmbedding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let shifted = [
            xs.shallow_clone(),
            xs.constant_pad_nd([0, 0, 1, -1]),
            xs.constant_pad_nd([0, 0, -1, 1]),
            xs.constant_pad_nd([1, -1, 0, 0]),
            xs.constant_pad_nd([-1, 1, 0, 0]),
        ];
        let shifted_refs: Vec<&Tensor> = shifted.iter().collect();
        let xs = Tensor::cat(&shifted_refs, 1);

        patchify_2d(&xs, self.patch_height, self.patch_width)
            .apply(&self.norm)
            .apply(&self.linear)
    }
}

#[derive(Debug)]
struct LocalSelfAttention {
    norm: nn::LayerNorm,
    to_qkv: nn::Linear,
    to_out: nn::Linear,
    temperature: Tensor,
    heads: i64,
}

impl LocalSelfAttention {
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
        let temperature = vs.var("temperature", &[], nn::Init::Const((dim_head as f64).powf(-0.5).ln()));

        Self {
            norm,
            to_qkv,
            to_out,
            temperature,
            heads,
        }
    }
}

impl nn::ModuleT for LocalSelfAttention {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        let qkv = xs.apply(&self.norm).apply(&self.to_qkv).chunk(3, -1);
        let q = split_heads(&qkv[0], self.heads);
        let k = split_heads(&qkv[1], self.heads);
        let v = split_heads(&qkv[2], self.heads);
        let tokens = xs.size()[1];
        let mask = Tensor::eye(tokens, (tch::Kind::Bool, xs.device()))
            .unsqueeze(0)
            .unsqueeze(0);
        let dots = (q.matmul(&k.transpose(-1, -2)) * self.temperature.exp())
            .masked_fill(&mask, f64::NEG_INFINITY);
        let attn = dots.softmax(-1, dots.kind());

        merge_heads(&attn.matmul(&v)).apply(&self.to_out)
    }
}

#[derive(Debug)]
struct SmallDatasetTransformerLayer {
    attention: LocalSelfAttention,
    feed_forward: FeedForward,
}

impl SmallDatasetTransformerLayer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        dropout: f64,
    ) -> Self {
        let attention = LocalSelfAttention::new(&(vs / "attention"), dim, heads, dim_head, dropout);
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), dim, mlp_dim, dropout);

        Self {
            attention,
            feed_forward,
        }
    }
}

impl nn::ModuleT for SmallDatasetTransformerLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + self.attention.forward_t(xs, train);
        let ff_out = self.feed_forward.forward_t(&xs, train);

        xs + ff_out
    }
}

#[derive(Debug)]
struct SmallDatasetTransformer {
    layers: Vec<SmallDatasetTransformerLayer>,
}

impl SmallDatasetTransformer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        depth: usize,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        dropout: f64,
    ) -> Self {
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                SmallDatasetTransformerLayer::new(
                    &(vs / name.as_str()),
                    dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    dropout,
                )
            })
            .collect();

        Self { layers }
    }
}

impl nn::ModuleT for SmallDatasetTransformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs
    }
}

#[derive(Debug)]
pub struct ViTForSmallDataset {
    patch_embedding: ShiftedPatchEmbedding,
    cls_token: Tensor,
    pos_embedding: Tensor,
    transformer: SmallDatasetTransformer,
    mlp_norm: nn::LayerNorm,
    mlp_head: nn::Linear,
    pool: Pool,
    emb_dropout: f64,
}

impl ViTForSmallDataset {
    pub fn new(vs: &nn::Path, config: ViTConfig) -> Self {
        let image_height = config.image_size.height;
        let image_width = config.image_size.width;
        let patch_height = config.patch_size.height;
        let patch_width = config.patch_size.width;
        let num_patches = num_patches(image_height, image_width, patch_height, patch_width);
        let patch_embedding = ShiftedPatchEmbedding::new(
            &(vs / "patch_embedding"),
            config.channels,
            config.dim,
            patch_height,
            patch_width,
        );
        let cls_token = vs.var(
            "cls_token",
            &[1, 1, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let pos_embedding = vs.var(
            "pos_embedding",
            &[1, num_patches + 1, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let transformer = SmallDatasetTransformer::new(
            &(vs / "transformer"),
            config.dim,
            config.depth,
            config.heads,
            config.dim_head,
            config.mlp_dim,
            config.dropout,
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
            cls_token,
            pos_embedding,
            transformer,
            mlp_norm,
            mlp_head,
            pool: config.pool,
            emb_dropout: config.emb_dropout,
        }
    }
}

impl nn::ModuleT for ViTForSmallDataset {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.apply(&self.patch_embedding);
        let batch = xs.size()[0];
        let tokens = xs.size()[1];
        let cls_tokens = repeat_token(&self.cls_token, batch);
        xs = Tensor::cat(&[cls_tokens, xs], 1);
        xs = (xs + self.pos_embedding.i((.., 0..tokens + 1, ..))).dropout(self.emb_dropout, train);
        let xs = self.transformer.forward_t(&xs, train);
        let pooled = match self.pool {
            Pool::Mean => xs.mean_dim(1, false, xs.kind()),
            Pool::Cls => xs.i((.., 0)),
        };

        pooled.apply(&self.mlp_norm).apply(&self.mlp_head)
    }
}
