use tch::{Tensor, nn};

use crate::{
    config::ViTConfig,
    layers::Transformer,
    models::vit::PatchEmbedding,
    positional::posemb_sincos_2d,
    tensor::{assert_image_patchable, patch_dim},
    token::PatchDropout,
};

#[derive(Debug)]
pub struct SimpleViTWithPatchDropout {
    patch_embedding: PatchEmbedding,
    pos_embedding: Tensor,
    patch_dropout: PatchDropout,
    transformer: Transformer,
    linear_head: nn::Linear,
}

impl SimpleViTWithPatchDropout {
    pub fn new(vs: &nn::Path, config: ViTConfig, patch_dropout: f64) -> Self {
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
        let transformer = Transformer::new(
            &(vs / "transformer"),
            config.dim,
            config.depth,
            config.heads,
            config.dim_head,
            config.mlp_dim,
            0.0,
        );
        let linear_head = nn::linear(
            vs / "linear_head",
            config.dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            patch_embedding,
            pos_embedding,
            patch_dropout: PatchDropout::new(patch_dropout),
            transformer,
            linear_head,
        }
    }
}

impl nn::ModuleT for SimpleViTWithPatchDropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let tokens = xs.apply(&self.patch_embedding) + self.pos_embedding.unsqueeze(0);
        let tokens = self.patch_dropout.forward_t(&tokens, train);
        let tokens = self.transformer.forward_t(&tokens, train);
        let pooled = tokens.mean_dim(1, false, tokens.kind());

        pooled.apply(&self.linear_head)
    }
}
