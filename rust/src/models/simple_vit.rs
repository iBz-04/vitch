use tch::{Tensor, nn, nn::ModuleT};

use crate::{
    config::ViTConfig,
    layers::Transformer,
    models::vit::PatchEmbedding,
    positional::posemb_sincos_2d,
    tensor::{assert_image_patchable, patch_dim},
};

#[derive(Debug)]
pub struct SimpleViT {
    patch_embedding: PatchEmbedding,
    pos_embedding: Tensor,
    transformer: Transformer,
    linear_head: nn::Linear,
}

impl SimpleViT {
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
            transformer,
            linear_head,
        }
    }

    pub fn patch_tokens(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.patch_embedding)
    }
}

impl nn::ModuleT for SimpleViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs.apply(&self.patch_embedding) + self.pos_embedding.unsqueeze(0);
        let xs = self.transformer.forward_t(&xs, train);
        let xs = xs.mean_dim(1, false, xs.kind());

        xs.apply(&self.linear_head)
    }
}
