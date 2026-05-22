use tch::{IndexOp, Tensor, nn};

use crate::{
    config::ViTConfig,
    conv_layers::DepthwiseConv2D,
    layers::Attention,
    models::vit::PatchEmbedding,
    tensor::{num_patches, patch_dim, repeat_token},
};

#[derive(Debug)]
struct LocalFeedForward {
    norm: nn::LayerNorm,
    conv1: nn::Conv2D,
    depthwise: DepthwiseConv2D,
    conv2: nn::Conv2D,
    dropout: f64,
}

impl LocalFeedForward {
    fn new(vs: &nn::Path, dim: i64, hidden_dim: i64, dropout: f64) -> Self {
        let norm = nn::layer_norm(vs / "norm", vec![dim], Default::default());
        let conv1 = nn::conv2d(vs / "conv1", dim, hidden_dim, 1, Default::default());
        let depthwise =
            DepthwiseConv2D::new(&(vs / "depthwise"), hidden_dim, hidden_dim, 3, 1, 1, true);
        let conv2 = nn::conv2d(vs / "conv2", hidden_dim, dim, 1, Default::default());

        Self {
            norm,
            conv1,
            depthwise,
            conv2,
            dropout,
        }
    }
}

impl nn::ModuleT for LocalFeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        let batch = size[0];
        let tokens = size[1];
        let dim = size[2];
        let side = (tokens as f64).sqrt() as i64;
        assert_eq!(
            side * side,
            tokens,
            "local feed-forward expects square token grid"
        );

        xs.apply(&self.norm)
            .view([batch, side, side, dim])
            .permute([0, 3, 1, 2])
            .apply(&self.conv1)
            .hardswish()
            .apply(&self.depthwise)
            .hardswish()
            .dropout(self.dropout, train)
            .apply(&self.conv2)
            .dropout(self.dropout, train)
            .permute([0, 2, 3, 1])
            .contiguous()
            .view([batch, tokens, dim])
    }
}

#[derive(Debug)]
struct LocalViTLayer {
    attention: Attention,
    feed_forward: LocalFeedForward,
}

impl LocalViTLayer {
    fn new(vs: &nn::Path, dim: i64, heads: i64, dim_head: i64, mlp_dim: i64, dropout: f64) -> Self {
        let attention = Attention::new(&(vs / "attention"), dim, heads, dim_head, dropout);
        let feed_forward = LocalFeedForward::new(&(vs / "feed_forward"), dim, mlp_dim, dropout);

        Self {
            attention,
            feed_forward,
        }
    }
}

impl nn::ModuleT for LocalViTLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + self.attention.forward_t(xs, train);
        let cls = xs.i((.., 0..1, ..));
        let tokens = xs.i((.., 1.., ..));
        let tokens_out = self.feed_forward.forward_t(&tokens, train);

        Tensor::cat(&[cls, tokens + tokens_out], 1)
    }
}

#[derive(Debug)]
struct LocalTransformer {
    layers: Vec<LocalViTLayer>,
}

impl LocalTransformer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        depth: usize,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        dropout: f64,
    ) -> Self {
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                LocalViTLayer::new(
                    &(vs / name.as_str()),
                    dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    dropout,
                )
            })
            .collect();

        Self { layers }
    }
}

impl nn::ModuleT for LocalTransformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs
    }
}

#[derive(Debug)]
pub struct LocalViT {
    patch_embedding: PatchEmbedding,
    cls_token: Tensor,
    pos_embedding: Tensor,
    transformer: LocalTransformer,
    mlp_norm: nn::LayerNorm,
    mlp_head: nn::Linear,
    emb_dropout: f64,
}

impl LocalViT {
    pub fn new(vs: &nn::Path, config: ViTConfig) -> Self {
        let image_height = config.image_size.height;
        let image_width = config.image_size.width;
        assert_eq!(
            image_height, image_width,
            "LocalViT expects square image size"
        );
        let patch_height = config.patch_size.height;
        let patch_width = config.patch_size.width;
        assert_eq!(
            patch_height, patch_width,
            "LocalViT expects square patch size"
        );
        let num_patches = num_patches(image_height, image_width, patch_height, patch_width);
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
        let transformer = LocalTransformer::new(
            &(vs / "transformer"),
            config.dim,
            config.depth,
            config.heads,
            config.dim_head,
            config.mlp_dim,
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
            cls_token,
            pos_embedding,
            transformer,
            mlp_norm,
            mlp_head,
            emb_dropout: config.emb_dropout,
        }
    }
}

impl nn::ModuleT for LocalViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.apply(&self.patch_embedding);
        let batch = xs.size()[0];
        let tokens = xs.size()[1];
        let cls_tokens = repeat_token(&self.cls_token, batch);
        xs = Tensor::cat(&[cls_tokens, xs], 1);
        xs = (xs + self.pos_embedding.i((.., 0..tokens + 1, ..))).dropout(self.emb_dropout, train);
        let xs = self.transformer.forward_t(&xs, train);
        let cls = xs.i((.., 0));

        cls.apply(&self.mlp_norm).apply(&self.mlp_head)
    }
}
