use tch::{IndexOp, Tensor, nn};

use crate::{
    config::{Pool, ViTConfig},
    layers::Transformer,
    tensor::{num_patches, patch_dim, patchify_2d, repeat_token},
};

#[derive(Debug)]
pub struct PatchEmbedding {
    norm1: nn::LayerNorm,
    linear: nn::Linear,
    norm2: nn::LayerNorm,
    patch_height: i64,
    patch_width: i64,
}

impl PatchEmbedding {
    pub fn new(
        vs: &nn::Path,
        patch_dim: i64,
        dim: i64,
        patch_height: i64,
        patch_width: i64,
    ) -> Self {
        let norm1 = nn::layer_norm(vs / "norm1", vec![patch_dim], Default::default());
        let linear = nn::linear(vs / "linear", patch_dim, dim, Default::default());
        let norm2 = nn::layer_norm(vs / "norm2", vec![dim], Default::default());

        Self {
            norm1,
            linear,
            norm2,
            patch_height,
            patch_width,
        }
    }
}

impl nn::Module for PatchEmbedding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        patchify_2d(xs, self.patch_height, self.patch_width)
            .apply(&self.norm1)
            .apply(&self.linear)
            .apply(&self.norm2)
    }
}

#[derive(Debug)]
pub struct ViT {
    patch_embedding: PatchEmbedding,
    cls_token: Tensor,
    pos_embedding: Tensor,
    transformer: Transformer,
    mlp_head: Option<nn::Linear>,
    pool: Pool,
    emb_dropout: f64,
}

impl ViT {
    pub fn new(vs: &nn::Path, config: ViTConfig) -> Self {
        let image_height = config.image_size.height;
        let image_width = config.image_size.width;
        let patch_height = config.patch_size.height;
        let patch_width = config.patch_size.width;
        let num_patches = num_patches(image_height, image_width, patch_height, patch_width);
        let patch_dim = patch_dim(config.channels, patch_height, patch_width);
        let num_cls_tokens = match config.pool {
            Pool::Cls => 1,
            Pool::Mean => 0,
        };
        let patch_embedding = PatchEmbedding::new(
            &(vs / "patch_embedding"),
            patch_dim,
            config.dim,
            patch_height,
            patch_width,
        );
        let cls_token = vs.var(
            "cls_token",
            &[num_cls_tokens, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let pos_embedding = vs.var(
            "pos_embedding",
            &[num_patches + num_cls_tokens, config.dim],
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
        let mlp_head = if config.num_classes > 0 {
            Some(nn::linear(
                vs / "mlp_head",
                config.dim,
                config.num_classes,
                Default::default(),
            ))
        } else {
            None
        };

        Self {
            patch_embedding,
            cls_token,
            pos_embedding,
            transformer,
            mlp_head,
            pool: config.pool,
            emb_dropout: config.emb_dropout,
        }
    }

    pub fn embeddings_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let batch = xs.size()[0];
        let mut xs = xs.apply(&self.patch_embedding);

        if self.pool == Pool::Cls {
            let cls_tokens = repeat_token(&self.cls_token, batch);
            xs = Tensor::cat(&[cls_tokens, xs], 1);
        }

        let seq_len = xs.size()[1];
        let pos_embedding = self.pos_embedding.i(0..seq_len).unsqueeze(0);
        xs = (xs + pos_embedding).dropout(self.emb_dropout, train);

        self.transformer.forward_t(&xs, train)
    }
}

impl nn::ModuleT for ViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = self.embeddings_t(xs, train);
        let xs = match self.pool {
            Pool::Mean => xs.mean_dim(1, false, xs.kind()),
            Pool::Cls => xs.i((.., 0)),
        };

        match &self.mlp_head {
            Some(mlp_head) => xs.apply(mlp_head),
            None => xs,
        }
    }
}
