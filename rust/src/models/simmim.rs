use tch::{Kind, Reduction, Tensor, nn};

use crate::{
    config::SimMIMConfig,
    layers::Transformer,
    masking::{RandomMask, gather_tokens},
    models::vit::PatchEmbedding,
    tensor::{num_patches, patch_dim, patchify_2d},
};

#[derive(Debug)]
pub struct SimMIM {
    patch_embedding: PatchEmbedding,
    pos_embedding: Tensor,
    encoder: Transformer,
    mask_token: Tensor,
    to_pixels: nn::Linear,
    masking_ratio: f64,
    patch_height: i64,
    patch_width: i64,
}

impl SimMIM {
    pub fn new(vs: &nn::Path, config: SimMIMConfig) -> Self {
        let encoder_config = config.encoder;
        let image_height = encoder_config.image_size.height;
        let image_width = encoder_config.image_size.width;
        let patch_height = encoder_config.patch_size.height;
        let patch_width = encoder_config.patch_size.width;
        let num_patches = num_patches(image_height, image_width, patch_height, patch_width);
        let patch_dim = patch_dim(encoder_config.channels, patch_height, patch_width);
        let patch_embedding = PatchEmbedding::new(
            &(vs / "patch_embedding"),
            patch_dim,
            encoder_config.dim,
            patch_height,
            patch_width,
        );
        let pos_embedding = vs.var(
            "pos_embedding",
            &[1, num_patches, encoder_config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let encoder = Transformer::new(
            &(vs / "encoder"),
            encoder_config.dim,
            encoder_config.depth,
            encoder_config.heads,
            encoder_config.dim_head,
            encoder_config.mlp_dim,
            encoder_config.dropout,
        );
        let mask_token = vs.var(
            "mask_token",
            &[encoder_config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let to_pixels = nn::linear(
            vs / "to_pixels",
            encoder_config.dim,
            patch_dim,
            Default::default(),
        );

        Self {
            patch_embedding,
            pos_embedding,
            encoder,
            mask_token,
            to_pixels,
            masking_ratio: config.masking_ratio,
            patch_height,
            patch_width,
        }
    }
}

impl nn::ModuleT for SimMIM {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let patches = patchify_2d(xs, self.patch_height, self.patch_width);
        let batch = patches.size()[0];
        let num_patches = patches.size()[1];
        let tokens = self.patch_embedding.forward_patches(&patches) + &self.pos_embedding;
        let mask = RandomMask::new(batch, num_patches, self.masking_ratio, xs.device());
        let mask_bool = Tensor::zeros([batch, num_patches], (Kind::Bool, xs.device()))
            .scatter_value(1, &mask.masked_indices, 1);
        let mask_tokens = self
            .mask_token
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat([batch, num_patches, 1])
            + &self.pos_embedding;
        let tokens = mask_bool.unsqueeze(-1).where_self(&mask_tokens, &tokens);
        let encoded = self.encoder.forward_t(&tokens, train);
        let encoded_mask_tokens = gather_tokens(&encoded, &mask.masked_indices);
        let pred_pixel_values = encoded_mask_tokens.apply(&self.to_pixels);
        let masked_patches = gather_tokens(&patches, &mask.masked_indices);
        let loss = pred_pixel_values.l1_loss(&masked_patches, Reduction::Mean);

        loss / (mask.masked_indices.size()[1] as f64)
    }
}
