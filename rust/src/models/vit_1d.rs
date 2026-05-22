use tch::{IndexOp, Tensor, nn, nn::ModuleT};

use crate::{
    config::ViT1DConfig,
    layers::Transformer,
    tensor::{assert_sequence_patchable, patchify_1d, repeat_token},
};

#[derive(Debug)]
pub struct PatchEmbedding1D {
    norm1: nn::LayerNorm,
    linear: nn::Linear,
    norm2: nn::LayerNorm,
    patch_size: i64,
}

impl PatchEmbedding1D {
    pub fn new(vs: &nn::Path, patch_dim: i64, dim: i64, patch_size: i64) -> Self {
        let norm1 = nn::layer_norm(vs / "norm1", vec![patch_dim], Default::default());
        let linear = nn::linear(vs / "linear", patch_dim, dim, Default::default());
        let norm2 = nn::layer_norm(vs / "norm2", vec![dim], Default::default());

        Self {
            norm1,
            linear,
            norm2,
            patch_size,
        }
    }
}

impl nn::Module for PatchEmbedding1D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        patchify_1d(xs, self.patch_size)
            .apply(&self.norm1)
            .apply(&self.linear)
            .apply(&self.norm2)
    }
}

#[derive(Debug)]
pub struct ViT1D {
    patch_embedding: PatchEmbedding1D,
    cls_token: Tensor,
    pos_embedding: Tensor,
    transformer: Transformer,
    mlp_norm: nn::LayerNorm,
    mlp_head: nn::Linear,
    emb_dropout: f64,
}

impl ViT1D {
    pub fn new(vs: &nn::Path, config: ViT1DConfig) -> Self {
        assert_sequence_patchable(config.seq_len, config.patch_size);
        let num_patches = config.seq_len / config.patch_size;
        let patch_dim = config.channels * config.patch_size;
        let patch_embedding =
            PatchEmbedding1D::new(&(vs / "patch_embedding"), patch_dim, config.dim, config.patch_size);
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
            &[config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let transformer = Transformer::new(
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
            emb_dropout: config.emb_dropout,
        }
    }
}

impl nn::ModuleT for ViT1D {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.apply(&self.patch_embedding);
        let batch = xs.size()[0];
        let tokens = xs.size()[1];
        let cls_tokens = repeat_token(&self.cls_token, batch);
        xs = Tensor::cat(&[cls_tokens, xs], 1);
        xs = (xs + self.pos_embedding.i((.., 0..tokens + 1, ..))).dropout(self.emb_dropout, train);
        let xs = self.transformer.forward_t(&xs, train).i((.., 0));

        xs.apply(&self.mlp_norm).apply(&self.mlp_head)
    }
}
