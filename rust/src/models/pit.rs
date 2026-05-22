use tch::{IndexOp, Tensor, nn};

use crate::{
    config::PiTConfig, conv_layers::DepthwiseConv2D, layers::Transformer, tensor::repeat_token,
};

#[derive(Debug)]
struct OverlapPatchEmbedding {
    linear: nn::Linear,
    patch_size: i64,
    stride: i64,
}

impl OverlapPatchEmbedding {
    fn new(vs: &nn::Path, channels: i64, dim: i64, patch_size: i64) -> Self {
        assert!(patch_size > 1, "patch size must be greater than one");
        let patch_dim = channels * patch_size * patch_size;
        let linear = nn::linear(vs / "linear", patch_dim, dim, Default::default());

        Self {
            linear,
            patch_size,
            stride: patch_size / 2,
        }
    }

    fn output_side(&self, image_size: i64) -> i64 {
        ((image_size - self.patch_size) / self.stride) + 1
    }
}

impl nn::Module for OverlapPatchEmbedding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.im2col(
            [self.patch_size, self.patch_size],
            [1, 1],
            [0, 0],
            [self.stride, self.stride],
        )
        .transpose(1, 2)
        .apply(&self.linear)
    }
}

#[derive(Debug)]
struct PoolTokens {
    downsample: DepthwiseConv2D,
    cls_proj: nn::Linear,
}

impl PoolTokens {
    fn new(vs: &nn::Path, dim: i64) -> Self {
        let downsample = DepthwiseConv2D::new(&(vs / "downsample"), dim, dim * 2, 3, 1, 2, true);
        let cls_proj = nn::linear(vs / "cls_proj", dim, dim * 2, Default::default());

        Self {
            downsample,
            cls_proj,
        }
    }
}

impl nn::Module for PoolTokens {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let cls = xs.i((.., 0..1, ..)).apply(&self.cls_proj);
        let tokens = xs.i((.., 1.., ..));
        let size = tokens.size();
        let batch = size[0];
        let num_tokens = size[1];
        let dim = size[2];
        let side = (num_tokens as f64).sqrt() as i64;
        assert_eq!(
            side * side,
            num_tokens,
            "PiT pooling expects square token grid"
        );
        let tokens = tokens
            .view([batch, side, side, dim])
            .permute([0, 3, 1, 2])
            .apply(&self.downsample);
        let pooled_size = tokens.size();
        let channels = pooled_size[1];
        let height = pooled_size[2];
        let width = pooled_size[3];
        let tokens = tokens
            .view([batch, channels, height * width])
            .transpose(1, 2);

        Tensor::cat(&[cls, tokens], 1)
    }
}

#[derive(Debug)]
struct PiTStage {
    transformer: Transformer,
    pool: Option<PoolTokens>,
}

impl PiTStage {
    fn new(
        vs: &nn::Path,
        dim: i64,
        depth: usize,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        dropout: f64,
        pool: bool,
    ) -> Self {
        let transformer = Transformer::new(
            &(vs / "transformer"),
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
        );
        let pool = if pool {
            Some(PoolTokens::new(&(vs / "pool"), dim))
        } else {
            None
        };

        Self { transformer, pool }
    }
}

impl nn::ModuleT for PiTStage {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = self.transformer.forward_t(xs, train);

        match &self.pool {
            Some(pool) => xs.apply(pool),
            None => xs,
        }
    }
}

#[derive(Debug)]
pub struct PiT {
    patch_embedding: OverlapPatchEmbedding,
    cls_token: Tensor,
    pos_embedding: Tensor,
    stages: Vec<PiTStage>,
    mlp_norm: nn::LayerNorm,
    mlp_head: nn::Linear,
    emb_dropout: f64,
}

impl PiT {
    pub fn new(vs: &nn::Path, config: PiTConfig) -> Self {
        assert_eq!(
            config.depths.len(),
            config.heads.len(),
            "PiT depths and heads must have matching lengths"
        );
        assert!(!config.depths.is_empty(), "PiT requires at least one stage");
        assert_eq!(
            config.image_size % config.patch_size,
            0,
            "image dimensions must be divisible by patch size"
        );
        let patch_embedding = OverlapPatchEmbedding::new(
            &(vs / "patch_embedding"),
            config.channels,
            config.dim,
            config.patch_size,
        );
        let output_side = patch_embedding.output_side(config.image_size);
        let num_patches = output_side * output_side;
        let cls_token = vs.var(
            "cls_token",
            &[1, 1, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let pos_embedding = vs.var(
            "pos_embedding",
            &[1, num_patches + 1, config.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let mut dim = config.dim;
        let stage_count = config.depths.len();
        let stages = config
            .depths
            .iter()
            .zip(config.heads.iter())
            .enumerate()
            .map(|(index, (depth, heads))| {
                let name = format!("stage_{index}");
                let is_last = index + 1 == stage_count;
                let stage = PiTStage::new(
                    &(vs / name.as_str()),
                    dim,
                    *depth,
                    *heads,
                    config.dim_head,
                    config.mlp_dim,
                    config.dropout,
                    !is_last,
                );
                if !is_last {
                    dim *= 2;
                }

                stage
            })
            .collect();
        let mlp_norm = nn::layer_norm(vs / "mlp_norm", vec![dim], Default::default());
        let mlp_head = nn::linear(vs / "mlp_head", dim, config.num_classes, Default::default());

        Self {
            patch_embedding,
            cls_token,
            pos_embedding,
            stages,
            mlp_norm,
            mlp_head,
            emb_dropout: config.emb_dropout,
        }
    }
}

impl nn::ModuleT for PiT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.apply(&self.patch_embedding);
        let batch = xs.size()[0];
        let tokens = xs.size()[1];
        let cls_tokens = repeat_token(&self.cls_token, batch);
        xs = Tensor::cat(&[cls_tokens, xs], 1);
        xs = (xs + self.pos_embedding.i((.., 0..tokens + 1, ..))).dropout(self.emb_dropout, train);

        for stage in &self.stages {
            xs = stage.forward_t(&xs, train);
        }

        xs.i((.., 0)).apply(&self.mlp_norm).apply(&self.mlp_head)
    }
}
