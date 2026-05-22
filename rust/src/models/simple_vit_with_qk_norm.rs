use tch::{Tensor, nn};

use crate::{
    config::ViTConfig,
    layers::FeedForward,
    models::vit::PatchEmbedding,
    norm::RmsNormHeads,
    positional::posemb_sincos_2d,
    tensor::{assert_image_patchable, merge_heads, patch_dim, split_heads},
};

#[derive(Debug)]
struct QkNormAttention {
    norm: nn::LayerNorm,
    q_norm: RmsNormHeads,
    k_norm: RmsNormHeads,
    to_qkv: nn::Linear,
    to_out: nn::Linear,
    heads: i64,
}

impl QkNormAttention {
    fn new(vs: &nn::Path, dim: i64, heads: i64, dim_head: i64) -> Self {
        let inner_dim = heads * dim_head;
        let norm = nn::layer_norm(vs / "norm", vec![dim], Default::default());
        let q_norm = RmsNormHeads::new(&(vs / "q_norm"), heads, dim_head);
        let k_norm = RmsNormHeads::new(&(vs / "k_norm"), heads, dim_head);
        let to_qkv = nn::linear(
            vs / "to_qkv",
            dim,
            inner_dim * 3,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        );
        let to_out = nn::linear(
            vs / "to_out",
            inner_dim,
            dim,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        Self {
            norm,
            q_norm,
            k_norm,
            to_qkv,
            to_out,
            heads,
        }
    }
}

impl nn::ModuleT for QkNormAttention {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        let qkv = xs.apply(&self.norm).apply(&self.to_qkv).chunk(3, -1);
        let q = self.q_norm.forward(&split_heads(&qkv[0], self.heads));
        let k = self.k_norm.forward(&split_heads(&qkv[1], self.heads));
        let v = split_heads(&qkv[2], self.heads);
        let attn = q.matmul(&k.transpose(-1, -2)).softmax(-1, q.kind());

        merge_heads(&attn.matmul(&v)).apply(&self.to_out)
    }
}

#[derive(Debug)]
struct QkNormTransformerLayer {
    attention: QkNormAttention,
    feed_forward: FeedForward,
}

impl QkNormTransformerLayer {
    fn new(vs: &nn::Path, dim: i64, heads: i64, dim_head: i64, mlp_dim: i64) -> Self {
        let attention = QkNormAttention::new(&(vs / "attention"), dim, heads, dim_head);
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), dim, mlp_dim, 0.0);

        Self {
            attention,
            feed_forward,
        }
    }
}

impl nn::ModuleT for QkNormTransformerLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + self.attention.forward_t(xs, train);
        let ff_out = self.feed_forward.forward_t(&xs, train);
        xs + ff_out
    }
}

#[derive(Debug)]
struct QkNormTransformer {
    layers: Vec<QkNormTransformerLayer>,
    norm: nn::LayerNorm,
}

impl QkNormTransformer {
    fn new(vs: &nn::Path, dim: i64, depth: usize, heads: i64, dim_head: i64, mlp_dim: i64) -> Self {
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                QkNormTransformerLayer::new(&(vs / name.as_str()), dim, heads, dim_head, mlp_dim)
            })
            .collect();
        let norm = nn::layer_norm(vs / "norm", vec![dim], Default::default());

        Self { layers, norm }
    }
}

impl nn::ModuleT for QkNormTransformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs.apply(&self.norm)
    }
}

#[derive(Debug)]
pub struct SimpleViTWithQkNorm {
    patch_embedding: PatchEmbedding,
    pos_embedding: Tensor,
    transformer: QkNormTransformer,
    linear_head: nn::LayerNorm,
}

impl SimpleViTWithQkNorm {
    pub fn new(vs: &nn::Path, config: ViTConfig) -> Self {
        let image_height = config.image_size.height;
        let image_width = config.image_size.width;
        let patch_height = config.patch_size.height;
        let patch_width = config.patch_size.width;
        assert_image_patchable(image_height, image_width, patch_height, patch_width);
        let patch_dim = patch_dim(config.channels, patch_height, patch_width);
        let patch_embedding = PatchEmbedding::new(
            &(vs / "patch_embedding"),
            patch_dim,
            config.dim,
            patch_height,
            patch_width,
        );
        let pos_embedding = posemb_sincos_2d(
            image_height / patch_height,
            image_width / patch_width,
            config.dim,
            10000.0,
            vs.device(),
        );
        let transformer = QkNormTransformer::new(
            &(vs / "transformer"),
            config.dim,
            config.depth,
            config.heads,
            config.dim_head,
            config.mlp_dim,
        );
        let linear_head = nn::layer_norm(vs / "linear_head", vec![config.dim], Default::default());

        Self {
            patch_embedding,
            pos_embedding,
            transformer,
            linear_head,
        }
    }
}

impl nn::ModuleT for SimpleViTWithQkNorm {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs.apply(&self.patch_embedding) + self.pos_embedding.unsqueeze(0);
        let xs = self.transformer.forward_t(&xs, train);
        let xs = xs.mean_dim(1, false, xs.kind());

        xs.apply(&self.linear_head)
    }
}
