use tch::{IndexOp, Reduction, Tensor, nn, nn::ModuleT};

use crate::{
    config::{MAEConfig, Pool},
    layers::Transformer,
    masking::{RandomMask, gather_tokens, scatter_tokens},
    models::vit::PatchEmbedding,
    tensor::{num_patches, patch_dim, patchify_2d},
};

#[derive(Debug)]
pub struct MAE {
    patch_embedding: PatchEmbedding,
    encoder_pos_embedding: Tensor,
    encoder: Transformer,
    enc_to_dec: Option<nn::Linear>,
    mask_token: Tensor,
    decoder: Transformer,
    decoder_pos_embedding: nn::Embedding,
    to_pixels: nn::Linear,
    masking_ratio: f64,
    patch_height: i64,
    patch_width: i64,
    pool: Pool,
}

impl MAE {
    pub fn new(vs: &nn::Path, config: MAEConfig) -> Self {
        let encoder_config = config.encoder;
        let image_height = encoder_config.image_size.height;
        let image_width = encoder_config.image_size.width;
        let patch_height = encoder_config.patch_size.height;
        let patch_width = encoder_config.patch_size.width;
        let num_patches = num_patches(image_height, image_width, patch_height, patch_width);
        let patch_dim = patch_dim(encoder_config.channels, patch_height, patch_width);
        let cls_tokens = match encoder_config.pool {
            Pool::Cls => 1,
            Pool::Mean => 0,
        };
        let patch_embedding = PatchEmbedding::new(
            &(vs / "patch_embedding"),
            patch_dim,
            encoder_config.dim,
            patch_height,
            patch_width,
        );
        let encoder_pos_embedding = vs.var(
            "encoder_pos_embedding",
            &[num_patches + cls_tokens, encoder_config.dim],
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
        let enc_to_dec = if encoder_config.dim == config.decoder_dim {
            None
        } else {
            Some(nn::linear(
                vs / "enc_to_dec",
                encoder_config.dim,
                config.decoder_dim,
                Default::default(),
            ))
        };
        let mask_token = vs.var(
            "mask_token",
            &[config.decoder_dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let decoder = Transformer::new(
            &(vs / "decoder"),
            config.decoder_dim,
            config.decoder_depth,
            config.decoder_heads,
            config.decoder_dim_head,
            config.decoder_dim * 4,
            0.0,
        );
        let decoder_pos_embedding = nn::embedding(
            vs / "decoder_pos_embedding",
            num_patches,
            config.decoder_dim,
            Default::default(),
        );
        let to_pixels = nn::linear(
            vs / "to_pixels",
            config.decoder_dim,
            patch_dim,
            Default::default(),
        );

        Self {
            patch_embedding,
            encoder_pos_embedding,
            encoder,
            enc_to_dec,
            mask_token,
            decoder,
            decoder_pos_embedding,
            to_pixels,
            masking_ratio: config.masking_ratio,
            patch_height,
            patch_width,
            pool: encoder_config.pool,
        }
    }

    fn encode_tokens(&self, tokens: &Tensor, train: bool) -> Tensor {
        self.encoder.forward_t(tokens, train)
    }
}

impl nn::ModuleT for MAE {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let patches = patchify_2d(xs, self.patch_height, self.patch_width);
        let batch = patches.size()[0];
        let num_patches = patches.size()[1];
        let tokens = self.patch_embedding.forward_patches(&patches);
        let pos_embedding = match self.pool {
            Pool::Cls => self.encoder_pos_embedding.i(1..num_patches + 1).unsqueeze(0),
            Pool::Mean => self.encoder_pos_embedding.unsqueeze(0),
        };
        let tokens = tokens + pos_embedding;
        let mask = RandomMask::new(batch, num_patches, self.masking_ratio, xs.device());
        let unmasked_tokens = gather_tokens(&tokens, &mask.unmasked_indices);
        let masked_patches = gather_tokens(&patches, &mask.masked_indices);
        let encoded_tokens = self.encode_tokens(&unmasked_tokens, train);
        let decoder_tokens = match &self.enc_to_dec {
            Some(enc_to_dec) => encoded_tokens.apply(enc_to_dec),
            None => encoded_tokens,
        };
        let unmasked_decoder_tokens =
            decoder_tokens + mask.unmasked_indices.apply(&self.decoder_pos_embedding);
        let num_masked = mask.masked_indices.size()[1];
        let mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0).repeat([batch, num_masked, 1])
            + mask.masked_indices.apply(&self.decoder_pos_embedding);
        let decoder_dim = mask_tokens.size()[2];
        let decoder_tokens = scatter_tokens(
            batch,
            num_patches,
            decoder_dim,
            &mask.unmasked_indices,
            &unmasked_decoder_tokens,
        ) + scatter_tokens(
            batch,
            num_patches,
            decoder_dim,
            &mask.masked_indices,
            &mask_tokens,
        );
        let decoded_tokens = self.decoder.forward_t(&decoder_tokens, train);
        let masked_tokens = gather_tokens(&decoded_tokens, &mask.masked_indices);
        let pred_pixel_values = masked_tokens.apply(&self.to_pixels);

        pred_pixel_values.mse_loss(&masked_patches, Reduction::Mean)
    }
}
