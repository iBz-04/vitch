use tch::{Tensor, nn};

use crate::{
    config::ViT3DConfig, layers::Transformer, models::vit_3d::PatchEmbedding3D,
    positional::posemb_sincos_3d, tensor::assert_video_patchable,
};

#[derive(Debug)]
pub struct SimpleViT3D {
    patch_embedding: PatchEmbedding3D,
    transformer: Transformer,
    linear_head: nn::Linear,
}

impl SimpleViT3D {
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
        let patch_dim = config.channels * config.frame_patch_size * patch_height * patch_width;
        let patch_embedding = PatchEmbedding3D::new(
            &(vs / "patch_embedding"),
            patch_dim,
            config.dim,
            config.frame_patch_size,
            patch_height,
            patch_width,
            false,
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
            transformer,
            linear_head,
        }
    }
}

impl nn::ModuleT for SimpleViT3D {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs.apply(&self.patch_embedding);
        let frame_tokens = xs.size()[1];
        let height_tokens = xs.size()[2];
        let width_tokens = xs.size()[3];
        let dim = xs.size()[4];
        let pe = posemb_sincos_3d(
            frame_tokens,
            height_tokens,
            width_tokens,
            dim,
            10000.0,
            xs.device(),
        );
        let batch = xs.size()[0];
        let xs =
            xs.view([batch, frame_tokens * height_tokens * width_tokens, dim]) + pe.unsqueeze(0);
        let xs = self.transformer.forward_t(&xs, train);
        let xs = xs.mean_dim(1, false, xs.kind());

        xs.apply(&self.linear_head)
    }
}
