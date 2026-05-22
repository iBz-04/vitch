use tch::{Tensor, nn};

use crate::{
    config::ViTConfig,
    layers::Transformer,
    models::vit::PatchEmbedding,
    positional::posemb_sincos_2d,
    tensor::{assert_image_patchable, patch_dim, repeat_token},
};

#[derive(Debug)]
pub struct SimpleViTWithRegisterTokens {
    patch_embedding: PatchEmbedding,
    register_tokens: Tensor,
    pos_embedding: Tensor,
    transformer: Transformer,
    linear_head: nn::Linear,
}

impl SimpleViTWithRegisterTokens {
    pub fn new(vs: &nn::Path, config: ViTConfig, num_register_tokens: i64) -> Self {
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
        let register_tokens = vs.var(
            "register_tokens",
            &[num_register_tokens, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
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
            register_tokens,
            pos_embedding,
            transformer,
            linear_head,
        }
    }
}

impl nn::ModuleT for SimpleViTWithRegisterTokens {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let batch = xs.size()[0];
        let tokens = xs.apply(&self.patch_embedding) + self.pos_embedding.unsqueeze(0);
        let register_tokens = repeat_token(&self.register_tokens, batch);
        let seq = Tensor::cat(&[tokens.shallow_clone(), register_tokens], 1);
        let seq = self.transformer.forward_t(&seq, train);
        let tokens = seq.narrow(1, 0, tokens.size()[1]);
        let pooled = tokens.mean_dim(1, false, tokens.kind());

        pooled.apply(&self.linear_head)
    }
}
