use tch::{Tensor, nn};

use crate::{
    config::ViT1DConfig, layers::Transformer, models::vit_1d::PatchEmbedding1D,
    positional::posemb_sincos_1d, tensor::assert_sequence_patchable,
};

#[derive(Debug)]
pub struct SimpleViT1D {
    patch_embedding: PatchEmbedding1D,
    transformer: Transformer,
    linear_head: nn::Linear,
}

impl SimpleViT1D {
    pub fn new(vs: &nn::Path, config: ViT1DConfig) -> Self {
        assert_sequence_patchable(config.seq_len, config.patch_size);
        let patch_dim = config.channels * config.patch_size;
        let patch_embedding = PatchEmbedding1D::new(
            &(vs / "patch_embedding"),
            patch_dim,
            config.dim,
            config.patch_size,
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

impl nn::ModuleT for SimpleViT1D {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs.apply(&self.patch_embedding);
        let tokens = xs.size()[1];
        let dim = xs.size()[2];
        let pe = posemb_sincos_1d(tokens, dim, 10000.0, xs.device());
        let xs = self.transformer.forward_t(&(xs + pe.unsqueeze(0)), train);
        let xs = xs.mean_dim(1, false, xs.kind());

        xs.apply(&self.linear_head)
    }
}
