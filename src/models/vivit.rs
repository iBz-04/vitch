use tch::{Kind, Tensor, nn};

use crate::{
    config::ViViTConfig,
    layers::Transformer,
    tensor::{patch_dim, patchify_3d_flat},
};

#[derive(Debug)]
pub struct ViViT {
    to_patch_embedding: nn::Linear,
    pos_embedding: Tensor,
    transformer: Transformer,
    head: nn::Linear,
    frame_patch_size: i64,
    patch_height: i64,
    patch_width: i64,
    emb_dropout: f64,
}

impl ViViT {
    pub fn new(vs: &nn::Path, config: ViViTConfig) -> Self {
        let patch_height = config.image_patch_size.height;
        let patch_width = config.image_patch_size.width;
        let grid_t = config.frames / config.frame_patch_size;
        let grid_h = config.image_size.height / patch_height;
        let grid_w = config.image_size.width / patch_width;
        let patch_dim = patch_dim(
            config.channels * config.frame_patch_size,
            patch_height,
            patch_width,
        );
        let to_patch_embedding = nn::linear(
            vs / "to_patch_embedding",
            patch_dim,
            config.dim,
            Default::default(),
        );
        let pos_embedding = vs.var(
            "pos_embedding",
            &[grid_t * grid_h * grid_w, config.dim],
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
            to_patch_embedding,
            pos_embedding,
            transformer,
            head,
            frame_patch_size: config.frame_patch_size,
            patch_height,
            patch_width,
            emb_dropout: config.emb_dropout,
        }
    }
}

impl nn::ModuleT for ViViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let patches = patchify_3d_flat(
            xs,
            self.frame_patch_size,
            self.patch_height,
            self.patch_width,
        );
        let tokens = patches.apply(&self.to_patch_embedding)
            + self
                .pos_embedding
                .narrow(0, 0, patches.size()[1])
                .unsqueeze(0);
        let tokens = self
            .transformer
            .forward_t(&tokens.dropout(self.emb_dropout, train), train);

        tokens.mean_dim(1, false, Kind::Float).apply(&self.head)
    }
}
