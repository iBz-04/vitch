use tch::{IndexOp, Tensor, nn};

use crate::{
    config::{Pool, ViTConfig},
    layers::{Attention, FeedForward},
    models::vit_with_patch_dropout::LinearPatchEmbedding,
    tensor::{num_patches, patch_dim, repeat_token},
};

#[derive(Debug)]
struct ParallelLayer {
    attention_branches: Vec<Attention>,
    feed_forward_branches: Vec<FeedForward>,
}

impl ParallelLayer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        num_parallel_branches: usize,
        dropout: f64,
    ) -> Self {
        let attention_branches = (0..num_parallel_branches)
            .map(|index| {
                let name = format!("attention_{index}");
                Attention::new(&(vs / name.as_str()), dim, heads, dim_head, dropout)
            })
            .collect();
        let feed_forward_branches = (0..num_parallel_branches)
            .map(|index| {
                let name = format!("feed_forward_{index}");
                FeedForward::new(&(vs / name.as_str()), dim, mlp_dim, dropout)
            })
            .collect();

        Self {
            attention_branches,
            feed_forward_branches,
        }
    }
}

impl nn::ModuleT for ParallelLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let attn_sum = self
            .attention_branches
            .iter()
            .map(|attention| attention.forward_t(xs, train))
            .reduce(|acc, value| acc + value)
            .expect("at least one attention branch is required");
        let xs = xs + attn_sum;
        let ff_sum = self
            .feed_forward_branches
            .iter()
            .map(|feed_forward| feed_forward.forward_t(&xs, train))
            .reduce(|acc, value| acc + value)
            .expect("at least one feed-forward branch is required");

        xs + ff_sum
    }
}

#[derive(Debug)]
struct ParallelTransformer {
    layers: Vec<ParallelLayer>,
}

impl ParallelTransformer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        depth: usize,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        num_parallel_branches: usize,
        dropout: f64,
    ) -> Self {
        assert!(num_parallel_branches > 0);
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                ParallelLayer::new(
                    &(vs / name.as_str()),
                    dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    num_parallel_branches,
                    dropout,
                )
            })
            .collect();

        Self { layers }
    }
}

impl nn::ModuleT for ParallelTransformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs
    }
}

#[derive(Debug)]
pub struct ParallelViT {
    patch_embedding: LinearPatchEmbedding,
    pos_embedding: Tensor,
    cls_token: Tensor,
    transformer: ParallelTransformer,
    mlp_norm: nn::LayerNorm,
    mlp_head: nn::Linear,
    pool: Pool,
    emb_dropout: f64,
}

impl ParallelViT {
    pub fn new(vs: &nn::Path, config: ViTConfig, num_parallel_branches: usize) -> Self {
        let image_height = config.image_size.height;
        let image_width = config.image_size.width;
        let patch_height = config.patch_size.height;
        let patch_width = config.patch_size.width;
        let num_patches = num_patches(image_height, image_width, patch_height, patch_width);
        let patch_dim = patch_dim(config.channels, patch_height, patch_width);
        let patch_embedding = LinearPatchEmbedding::new(
            &(vs / "patch_embedding"),
            patch_dim,
            config.dim,
            patch_height,
            patch_width,
        );
        let pos_embedding = vs.var(
            "pos_embedding",
            &[1, num_patches + 1, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let cls_token = vs.var(
            "cls_token",
            &[1, 1, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let transformer = ParallelTransformer::new(
            &(vs / "transformer"),
            config.dim,
            config.depth,
            config.heads,
            config.dim_head,
            config.mlp_dim,
            num_parallel_branches,
            config.dropout,
        );
        let mlp_norm = nn::layer_norm(vs / "mlp_norm", vec![config.dim], Default::default());
        let mlp_head = nn::linear(
            vs / "mlp_head",
            config.dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            patch_embedding,
            pos_embedding,
            cls_token,
            transformer,
            mlp_norm,
            mlp_head,
            pool: config.pool,
            emb_dropout: config.emb_dropout,
        }
    }
}

impl nn::ModuleT for ParallelViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.apply(&self.patch_embedding);
        let batch = xs.size()[0];
        let tokens = xs.size()[1];
        let cls_tokens = repeat_token(&self.cls_token, batch);
        xs = Tensor::cat(&[cls_tokens, xs], 1);
        xs = (xs + self.pos_embedding.i((.., 0..tokens + 1, ..))).dropout(self.emb_dropout, train);
        let xs = self.transformer.forward_t(&xs, train);
        let pooled = match self.pool {
            Pool::Mean => xs.mean_dim(1, false, xs.kind()),
            Pool::Cls => xs.i((.., 0)),
        };

        pooled.apply(&self.mlp_norm).apply(&self.mlp_head)
    }
}
