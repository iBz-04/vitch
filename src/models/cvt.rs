use tch::{Kind, Tensor, nn};

use crate::{
    config::{CvTConfig, CvTStageConfig},
    conv_layers::{ChannelLayerNorm2D, DepthwiseConv2D},
};

#[derive(Debug)]
struct ConvFeedForward {
    norm: ChannelLayerNorm2D,
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    dropout: f64,
}

impl ConvFeedForward {
    fn new(vs: &nn::Path, dim: i64, mult: i64, dropout: f64) -> Self {
        assert!(mult > 0, "mlp multiplier must be positive");
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), dim, 1e-5);
        let conv1 = nn::conv2d(vs / "conv1", dim, dim * mult, 1, Default::default());
        let conv2 = nn::conv2d(vs / "conv2", dim * mult, dim, 1, Default::default());

        Self {
            norm,
            conv1,
            conv2,
            dropout,
        }
    }
}

impl nn::ModuleT for ConvFeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.norm)
            .apply(&self.conv1)
            .gelu("none")
            .dropout(self.dropout, train)
            .apply(&self.conv2)
            .dropout(self.dropout, train)
    }
}

#[derive(Debug)]
struct ConvAttention {
    norm: ChannelLayerNorm2D,
    to_q: DepthwiseConv2D,
    to_kv: DepthwiseConv2D,
    to_out: nn::Conv2D,
    heads: i64,
    dim_head: i64,
    scale: f64,
    dropout: f64,
}

impl ConvAttention {
    fn new(
        vs: &nn::Path,
        dim: i64,
        proj_kernel: i64,
        kv_proj_stride: i64,
        heads: i64,
        dim_head: i64,
        dropout: f64,
    ) -> Self {
        assert!(heads > 0, "heads must be positive");
        assert!(dim_head > 0, "dim head must be positive");
        assert!(proj_kernel > 0, "projection kernel must be positive");
        assert!(kv_proj_stride > 0, "kv projection stride must be positive");
        let inner_dim = heads * dim_head;
        let padding = proj_kernel / 2;
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), dim, 1e-5);
        let to_q = DepthwiseConv2D::new(
            &(vs / "to_q"),
            dim,
            inner_dim,
            proj_kernel,
            padding,
            1,
            false,
        );
        let to_kv = DepthwiseConv2D::new(
            &(vs / "to_kv"),
            dim,
            inner_dim * 2,
            proj_kernel,
            padding,
            kv_proj_stride,
            false,
        );
        let to_out = nn::conv2d(vs / "to_out", inner_dim, dim, 1, Default::default());

        Self {
            norm,
            to_q,
            to_kv,
            to_out,
            heads,
            dim_head,
            scale: (dim_head as f64).powf(-0.5),
            dropout,
        }
    }
}

impl nn::ModuleT for ConvAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        let batch = size[0];
        let height = size[2];
        let width = size[3];
        let xs = xs.apply(&self.norm);
        let q = xs.apply(&self.to_q);
        let kv = xs.apply(&self.to_kv).chunk(2, 1);
        let q = q
            .view([batch, self.heads, self.dim_head, height * width])
            .transpose(2, 3);
        let k_size = kv[0].size();
        let kv_tokens = k_size[2] * k_size[3];
        let k = kv[0]
            .view([batch, self.heads, self.dim_head, kv_tokens])
            .transpose(2, 3);
        let v = kv[1]
            .view([batch, self.heads, self.dim_head, kv_tokens])
            .transpose(2, 3);
        let attn = (q.matmul(&k.transpose(-1, -2)) * self.scale)
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train);

        attn.matmul(&v)
            .transpose(2, 3)
            .contiguous()
            .view([batch, self.heads * self.dim_head, height, width])
            .apply(&self.to_out)
            .dropout(self.dropout, train)
    }
}

#[derive(Debug)]
struct ConvTransformerLayer {
    attention: ConvAttention,
    feed_forward: ConvFeedForward,
}

impl ConvTransformerLayer {
    fn new(vs: &nn::Path, dim: i64, stage: CvTStageConfig, dim_head: i64, dropout: f64) -> Self {
        let attention = ConvAttention::new(
            &(vs / "attention"),
            dim,
            stage.proj_kernel,
            stage.kv_proj_stride,
            stage.heads,
            dim_head,
            dropout,
        );
        let feed_forward =
            ConvFeedForward::new(&(vs / "feed_forward"), dim, stage.mlp_mult, dropout);

        Self {
            attention,
            feed_forward,
        }
    }
}

impl nn::ModuleT for ConvTransformerLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + self.attention.forward_t(xs, train);
        let ff_out = self.feed_forward.forward_t(&xs, train);

        xs + ff_out
    }
}

#[derive(Debug)]
struct CvTStage {
    embedding: nn::Conv2D,
    norm: ChannelLayerNorm2D,
    layers: Vec<ConvTransformerLayer>,
}

impl CvTStage {
    fn new(vs: &nn::Path, dim_in: i64, stage: CvTStageConfig, dim_head: i64, dropout: f64) -> Self {
        assert!(stage.embed_kernel > 0, "embedding kernel must be positive");
        assert!(stage.embed_stride > 0, "embedding stride must be positive");
        let embedding = nn::conv2d(
            vs / "embedding",
            dim_in,
            stage.embed_dim,
            stage.embed_kernel,
            nn::ConvConfig {
                stride: stage.embed_stride,
                padding: stage.embed_kernel / 2,
                ..Default::default()
            },
        );
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), stage.embed_dim, 1e-5);
        let layers = (0..stage.depth)
            .map(|index| {
                let name = format!("layer_{index}");
                ConvTransformerLayer::new(
                    &(vs / name.as_str()),
                    stage.embed_dim,
                    stage,
                    dim_head,
                    dropout,
                )
            })
            .collect();

        Self {
            embedding,
            norm,
            layers,
        }
    }
}

impl nn::ModuleT for CvTStage {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.apply(&self.embedding).apply(&self.norm);

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs
    }
}

#[derive(Debug)]
pub struct CvT {
    stages: Vec<CvTStage>,
    head: nn::Linear,
}

impl CvT {
    pub fn new(vs: &nn::Path, config: CvTConfig) -> Self {
        let mut dim_in = config.channels;
        let stages = config
            .stages
            .iter()
            .enumerate()
            .map(|(index, stage)| {
                let name = format!("stage_{index}");
                let built = CvTStage::new(
                    &(vs / name.as_str()),
                    dim_in,
                    *stage,
                    config.dim_head,
                    config.dropout,
                );
                dim_in = stage.embed_dim;

                built
            })
            .collect();
        let head = nn::linear(vs / "head", dim_in, config.num_classes, Default::default());

        Self { stages, head }
    }
}

impl nn::ModuleT for CvT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for stage in &self.stages {
            xs = stage.forward_t(&xs, train);
        }

        xs.adaptive_avg_pool2d([1, 1]).flat_view().apply(&self.head)
    }
}
