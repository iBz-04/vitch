use tch::{IndexOp, Tensor, nn};

use crate::{
    config::{Pool, ViT3DConfig},
    layers::Transformer,
    tensor::{assert_video_patchable, patchify_3d_flat, repeat_token},
};

#[derive(Debug)]
pub struct PatchEmbedding3D {
    norm1: nn::LayerNorm,
    linear: nn::Linear,
    norm2: nn::LayerNorm,
    frame_patch_size: i64,
    patch_height: i64,
    patch_width: i64,
    flatten: bool,
}

impl PatchEmbedding3D {
    pub fn new(
        vs: &nn::Path,
        patch_dim: i64,
        dim: i64,
        frame_patch_size: i64,
        patch_height: i64,
        patch_width: i64,
        flatten: bool,
    ) -> Self {
        let norm1 = nn::layer_norm(vs / "norm1", vec![patch_dim], Default::default());
        let linear = nn::linear(vs / "linear", patch_dim, dim, Default::default());
        let norm2 = nn::layer_norm(vs / "norm2", vec![dim], Default::default());

        Self {
            norm1,
            linear,
            norm2,
            frame_patch_size,
            patch_height,
            patch_width,
            flatten,
        }
    }
}

impl nn::Module for PatchEmbedding3D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = if self.flatten {
            patchify_3d_flat(
                xs,
                self.frame_patch_size,
                self.patch_height,
                self.patch_width,
            )
        } else {
            crate::tensor::patchify_3d_grid(
                xs,
                self.frame_patch_size,
                self.patch_height,
                self.patch_width,
            )
        };

        xs.apply(&self.norm1).apply(&self.linear).apply(&self.norm2)
    }
}

#[derive(Debug)]
pub struct ViT3D {
    patch_embedding: PatchEmbedding3D,
    cls_token: Tensor,
    pos_embedding: Tensor,
    transformer: Transformer,
    mlp_norm: nn::LayerNorm,
    mlp_head: nn::Linear,
    pool: Pool,
    emb_dropout: f64,
}

impl ViT3D {
    pub fn new(vs: &nn::Path, config: ViT3DConfig) -> Self {
        let image_height = config.image_size.height;
        let image_width = config.image_size.width;
        let patch_height = config.image_patch_size.height;
        let patch_width = config.image_patch_size.width;
        assert_video_patchable(
            config.frames,
            config.frame_patch_size,
            image_height,
            image_width,
            patch_height,
            patch_width,
        );
        let frame_tokens = config.frames / config.frame_patch_size;
        let height_tokens = image_height / patch_height;
        let width_tokens = image_width / patch_width;
        let num_patches = frame_tokens * height_tokens * width_tokens;
        let patch_dim = config.channels * config.frame_patch_size * patch_height * patch_width;
        let patch_embedding = PatchEmbedding3D::new(
            &(vs / "patch_embedding"),
            patch_dim,
            config.dim,
            config.frame_patch_size,
            patch_height,
            patch_width,
            true,
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
            pool: config.pool,
            emb_dropout: config.emb_dropout,
        }
    }
}

impl nn::ModuleT for ViT3D {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.apply(&self.patch_embedding);
        let batch = xs.size()[0];
        let tokens = xs.size()[1];
        let cls_tokens = repeat_token(&self.cls_token, batch);
        xs = Tensor::cat(&[cls_tokens, xs], 1);
        xs = (xs + self.pos_embedding.i((.., 0..tokens + 1, ..))).dropout(self.emb_dropout, train);
        let xs = self.transformer.forward_t(&xs, train);
        let xs = match self.pool {
            Pool::Mean => xs.mean_dim(1, false, xs.kind()),
            Pool::Cls => xs.i((.., 0)),
        };

        xs.apply(&self.mlp_norm).apply(&self.mlp_head)
    }
}
