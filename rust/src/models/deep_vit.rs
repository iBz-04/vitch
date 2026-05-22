use tch::{IndexOp, Tensor, nn};

use crate::{
    config::{Pool, ViTConfig},
    layers::FeedForward,
    models::vit::PatchEmbedding,
    tensor::{merge_heads, num_patches, patch_dim, repeat_token, split_heads},
};

#[derive(Debug)]
struct ReAttention {
    norm: nn::LayerNorm,
    to_qkv: nn::Linear,
    reattn_weights: Tensor,
    reattn_norm: nn::LayerNorm,
    to_out: nn::Linear,
    heads: i64,
    scale: f64,
    dropout: f64,
}

impl ReAttention {
    fn new(vs: &nn::Path, dim: i64, heads: i64, dim_head: i64, dropout: f64) -> Self {
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
        let reattn_weights = vs.var(
            "reattn_weights",
            &[heads, heads],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let reattn_norm = nn::layer_norm(vs / "reattn_norm", vec![heads], Default::default());
        let to_out = nn::linear(vs / "to_out", inner_dim, dim, Default::default());

        Self {
            norm,
            to_qkv,
            reattn_weights,
            reattn_norm,
            to_out,
            heads,
            scale: (dim_head as f64).powf(-0.5),
            dropout,
        }
    }
}

impl nn::ModuleT for ReAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let qkv = xs.apply(&self.norm).apply(&self.to_qkv).chunk(3, -1);
        let q = split_heads(&qkv[0], self.heads);
        let k = split_heads(&qkv[1], self.heads);
        let v = split_heads(&qkv[2], self.heads);
        let attn = (q.matmul(&k.transpose(-1, -2)) * self.scale)
            .softmax(-1, q.kind())
            .dropout(self.dropout, train);
        let attn = attn
            .permute([0, 2, 3, 1])
            .matmul(&self.reattn_weights)
            .permute([0, 3, 1, 2]);
        let attn = attn
            .permute([0, 2, 3, 1])
            .apply(&self.reattn_norm)
            .permute([0, 3, 1, 2]);

        merge_heads(&attn.matmul(&v))
            .apply(&self.to_out)
            .dropout(self.dropout, train)
    }
}

#[derive(Debug)]
struct DeepViTLayer {
    attention: ReAttention,
    feed_forward: FeedForward,
}

impl DeepViTLayer {
    fn new(vs: &nn::Path, dim: i64, heads: i64, dim_head: i64, mlp_dim: i64, dropout: f64) -> Self {
        let attention = ReAttention::new(&(vs / "attention"), dim, heads, dim_head, dropout);
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), dim, mlp_dim, dropout);

        Self {
            attention,
            feed_forward,
        }
    }
}

impl nn::ModuleT for DeepViTLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + self.attention.forward_t(xs, train);
        let ff_out = self.feed_forward.forward_t(&xs, train);
        xs + ff_out
    }
}

#[derive(Debug)]
struct DeepViTTransformer {
    layers: Vec<DeepViTLayer>,
}

impl DeepViTTransformer {
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
                DeepViTLayer::new(
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

impl nn::ModuleT for DeepViTTransformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs
    }
}

#[derive(Debug)]
pub struct DeepViT {
    patch_embedding: PatchEmbedding,
    cls_token: Tensor,
    pos_embedding: Tensor,
    transformer: DeepViTTransformer,
    mlp_norm: nn::LayerNorm,
    mlp_head: nn::Linear,
    pool: Pool,
    emb_dropout: f64,
}

impl DeepViT {
    pub fn new(vs: &nn::Path, config: ViTConfig) -> Self {
        let image_height = config.image_size.height;
        let image_width = config.image_size.width;
        assert_eq!(
            image_height, image_width,
            "DeepViT expects square image size"
        );
        let patch_height = config.patch_size.height;
        let patch_width = config.patch_size.width;
        assert_eq!(
            patch_height, patch_width,
            "DeepViT expects square patch size"
        );
        let num_patches = num_patches(image_height, image_width, patch_height, patch_width);
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
            &[1, num_patches + 1, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let cls_token = vs.var(
            "cls_token",
            &[1, 1, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let transformer = DeepViTTransformer::new(
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

impl nn::ModuleT for DeepViT {
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
