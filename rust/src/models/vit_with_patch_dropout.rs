use tch::{IndexOp, Tensor, nn};

use crate::{
    config::{Pool, ViTConfig},
    layers::Transformer,
    tensor::{num_patches, patch_dim, patchify_2d, repeat_token},
    token::PatchDropout,
};

#[derive(Debug)]
pub struct LinearPatchEmbedding {
    linear: nn::Linear,
    patch_height: i64,
    patch_width: i64,
}

impl LinearPatchEmbedding {
    pub fn new(
        vs: &nn::Path,
        patch_dim: i64,
        dim: i64,
        patch_height: i64,
        patch_width: i64,
    ) -> Self {
        let linear = nn::linear(vs / "linear", patch_dim, dim, Default::default());

        Self {
            linear,
            patch_height,
            patch_width,
        }
    }
}

impl nn::Module for LinearPatchEmbedding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        patchify_2d(xs, self.patch_height, self.patch_width).apply(&self.linear)
    }
}

#[derive(Debug)]
pub struct ViTWithPatchDropout {
    patch_embedding: LinearPatchEmbedding,
    pos_embedding: Tensor,
    cls_token: Tensor,
    patch_dropout: PatchDropout,
    transformer: Transformer,
    mlp_norm: nn::LayerNorm,
    mlp_head: nn::Linear,
    pool: Pool,
    emb_dropout: f64,
}

impl ViTWithPatchDropout {
    pub fn new(vs: &nn::Path, config: ViTConfig, patch_dropout: f64) -> Self {
        let image_height = config.image_size.height;
        let image_width = config.image_size.width;
        let patch_height = config.patch_size.height;
        let patch_width = config.patch_size.width;
        let num_patches = num_patches(image_height, image_width, patch_height, patch_width);
        let patch_dim = patch_dim(config.channels, patch_height, patch_width);
        let patch_embedding = LinearPatchEmbedding::new(
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
            &[1, 1, config.dim],
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
            pos_embedding,
            cls_token,
            patch_dropout: PatchDropout::new(patch_dropout),
            transformer,
            mlp_norm,
            mlp_head,
            pool: config.pool,
            emb_dropout: config.emb_dropout,
        }
    }
}

impl nn::ModuleT for ViTWithPatchDropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let batch = xs.size()[0];
        let tokens = xs.apply(&self.patch_embedding) + self.pos_embedding.unsqueeze(0);
        let tokens = self.patch_dropout.forward_t(&tokens, train);
        let cls_tokens = repeat_token(&self.cls_token, batch);
        let tokens = Tensor::cat(&[cls_tokens, tokens], 1).dropout(self.emb_dropout, train);
        let tokens = self.transformer.forward_t(&tokens, train);
        let pooled = match self.pool {
            Pool::Mean => tokens.mean_dim(1, false, tokens.kind()),
            Pool::Cls => tokens.i((.., 0)),
        };

        pooled.apply(&self.mlp_norm).apply(&self.mlp_head)
    }
}
