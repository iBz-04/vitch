use tch::{IndexOp, Kind, Tensor, nn, nn::ModuleT};

use crate::{
    config::{CompactVisionConfig, Pool},
    layers::{MaskedAttention, Transformer},
    models::vit::PatchEmbedding,
    tensor::{num_patches, patch_dim, patchify_2d, repeat_token},
};

#[derive(Debug)]
pub(crate) struct CompactImageTransformer {
    patch_embedding: PatchEmbedding,
    cls_token: Tensor,
    pos_embedding: Tensor,
    transformer: Transformer,
    head: nn::Linear,
    pool: Pool,
    emb_dropout: f64,
}

impl CompactImageTransformer {
    pub(crate) fn new(vs: &nn::Path, config: CompactVisionConfig) -> Self {
        let patch_height = config.patch_size.height;
        let patch_width = config.patch_size.width;
        let num_patches = num_patches(
            config.image_size.height,
            config.image_size.width,
            patch_height,
            patch_width,
        );
        let patch_dim = patch_dim(config.channels, patch_height, patch_width);
        let cls_tokens = match config.pool {
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
            &[cls_tokens, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let pos_embedding = vs.var(
            "pos_embedding",
            &[num_patches + cls_tokens, config.dim],
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
        let head = nn::linear(
            vs / "head",
            config.dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            patch_embedding,
            cls_token,
            pos_embedding,
            transformer,
            head,
            pool: config.pool,
            emb_dropout: config.emb_dropout,
        }
    }

    pub(crate) fn tokens_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let batch = xs.size()[0];
        let mut tokens = xs.apply(&self.patch_embedding);

        if self.pool == Pool::Cls {
            tokens = Tensor::cat(&[repeat_token(&self.cls_token, batch), tokens], 1);
        }

        let tokens_len = tokens.size()[1];
        let pos = self.pos_embedding.i(0..tokens_len).unsqueeze(0);

        self.transformer
            .forward_t(&(tokens + pos).dropout(self.emb_dropout, train), train)
    }
}

impl nn::ModuleT for CompactImageTransformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let tokens = self.tokens_t(xs, train);
        let pooled = match self.pool {
            Pool::Cls => tokens.i((.., 0)),
            Pool::Mean => tokens.mean_dim(1, false, tokens.kind()),
        };

        pooled.apply(&self.head)
    }
}

#[derive(Debug)]
pub(crate) struct MaskedImageTransformer {
    patch_embedding: PatchEmbedding,
    pos_embedding: Tensor,
    attention_layers: Vec<(MaskedAttention, nn::LayerNorm, nn::Linear, nn::Linear)>,
    head: nn::Linear,
    patch_height: i64,
    patch_width: i64,
}

impl MaskedImageTransformer {
    pub(crate) fn new(vs: &nn::Path, config: CompactVisionConfig) -> Self {
        let patch_height = config.patch_size.height;
        let patch_width = config.patch_size.width;
        let num_patches = num_patches(
            config.image_size.height,
            config.image_size.width,
            patch_height,
            patch_width,
        );
        let patch_dim = patch_dim(config.channels, patch_height, patch_width);
        let patch_embedding = PatchEmbedding::new(
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
        let attention_layers = (0..config.depth)
            .map(|index| {
                let name = format!("layer_{index}");
                let layer_vs = vs / name.as_str();
                let attn = MaskedAttention::new(
                    &(layer_vs.clone() / "attention"),
                    config.dim,
                    config.heads,
                    config.dim_head,
                    config.dropout,
                );
                let norm = nn::layer_norm(
                    layer_vs.clone() / "ff_norm",
                    vec![config.dim],
                    Default::default(),
                );
                let fc1 = nn::linear(
                    layer_vs.clone() / "ff1",
                    config.dim,
                    config.mlp_dim,
                    Default::default(),
                );
                let fc2 = nn::linear(
                    layer_vs / "ff2",
                    config.mlp_dim,
                    config.dim,
                    Default::default(),
                );
                (attn, norm, fc1, fc2)
            })
            .collect();
        let head = nn::linear(
            vs / "head",
            config.dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            patch_embedding,
            pos_embedding,
            attention_layers,
            head,
            patch_height,
            patch_width,
        }
    }

    pub(crate) fn patch_grid_for(&self, xs: &Tensor) -> (i64, i64) {
        let size = xs.size();
        assert_eq!(
            size.len(),
            4,
            "expected image tensor with shape [batch, channels, height, width]"
        );
        (size[2] / self.patch_height, size[3] / self.patch_width)
    }

    pub(crate) fn forward_t_with_mask(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        train: bool,
    ) -> Tensor {
        let patches = patchify_2d(xs, self.patch_height, self.patch_width);
        let mut tokens = self.patch_embedding.forward_patches(&patches) + &self.pos_embedding;

        for (attn, norm, fc1, fc2) in &self.attention_layers {
            let attn_out = attn.forward_t_with_mask(&tokens, mask, train);
            tokens += attn_out;
            let ff_out = tokens
                .apply(norm)
                .apply(fc1)
                .gelu("none")
                .dropout(0.0, train)
                .apply(fc2);
            tokens += ff_out;
        }

        tokens.mean_dim(1, false, Kind::Float).apply(&self.head)
    }
}
