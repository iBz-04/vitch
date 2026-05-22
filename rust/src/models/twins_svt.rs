use tch::{Kind, Tensor, nn};

use crate::{
    config::{TwinsSVTConfig, TwinsSVTStageConfig},
    conv_layers::ChannelLayerNorm2D,
    tensor::{assert_image_patchable, window_partition, window_unpartition},
};

#[derive(Debug)]
struct PatchEmbedding {
    patch_size: i64,
    norm1: ChannelLayerNorm2D,
    proj: nn::Conv2D,
    norm2: ChannelLayerNorm2D,
}

impl PatchEmbedding {
    fn new(vs: &nn::Path, dim: i64, dim_out: i64, patch_size: i64) -> Self {
        assert!(patch_size > 0, "patch size must be positive");
        let patch_dim = dim * patch_size * patch_size;
        let norm1 = ChannelLayerNorm2D::new(&(vs / "norm1"), patch_dim, 1e-5);
        let proj = nn::conv2d(vs / "proj", patch_dim, dim_out, 1, Default::default());
        let norm2 = ChannelLayerNorm2D::new(&(vs / "norm2"), dim_out, 1e-5);

        Self {
            patch_size,
            norm1,
            proj,
            norm2,
        }
    }
}

impl nn::ModuleT for PatchEmbedding {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        let size = xs.size();
        let batch = size[0];
        let channels = size[1];
        let height = size[2];
        let width = size[3];
        assert_image_patchable(height, width, self.patch_size, self.patch_size);
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;

        xs.view([
            batch,
            channels,
            grid_h,
            self.patch_size,
            grid_w,
            self.patch_size,
        ])
        .permute([0, 1, 3, 5, 2, 4])
        .contiguous()
        .view([
            batch,
            channels * self.patch_size * self.patch_size,
            grid_h,
            grid_w,
        ])
        .apply(&self.norm1)
        .apply(&self.proj)
        .apply(&self.norm2)
    }
}

#[derive(Debug)]
struct FeedForward2D {
    norm: ChannelLayerNorm2D,
    fc1: nn::Conv2D,
    fc2: nn::Conv2D,
    dropout: f64,
}

impl FeedForward2D {
    fn new(vs: &nn::Path, dim: i64, mult: i64, dropout: f64) -> Self {
        assert!(mult > 0, "mlp multiplier must be positive");
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), dim, 1e-5);
        let fc1 = nn::conv2d(vs / "fc1", dim, dim * mult, 1, Default::default());
        let fc2 = nn::conv2d(vs / "fc2", dim * mult, dim, 1, Default::default());

        Self {
            norm,
            fc1,
            fc2,
            dropout,
        }
    }
}

impl nn::ModuleT for FeedForward2D {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.norm)
            .apply(&self.fc1)
            .gelu("none")
            .dropout(self.dropout, train)
            .apply(&self.fc2)
            .dropout(self.dropout, train)
    }
}

#[derive(Debug)]
struct LocalAttention {
    norm: ChannelLayerNorm2D,
    to_q: nn::Conv2D,
    to_kv: nn::Conv2D,
    to_out: nn::Conv2D,
    heads: i64,
    dim_head: i64,
    patch_size: i64,
    scale: f64,
    dropout: f64,
}

impl LocalAttention {
    fn new(
        vs: &nn::Path,
        dim: i64,
        heads: i64,
        dim_head: i64,
        patch_size: i64,
        dropout: f64,
    ) -> Self {
        assert!(patch_size > 0, "local patch size must be positive");
        assert!(heads > 0, "heads must be positive");
        assert!(dim_head > 0, "dim head must be positive");
        let inner_dim = heads * dim_head;
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), dim, 1e-5);
        let conv_config = nn::ConvConfig {
            bias: false,
            ..Default::default()
        };
        let to_q = nn::conv2d(vs / "to_q", dim, inner_dim, 1, conv_config);
        let to_kv = nn::conv2d(vs / "to_kv", dim, inner_dim * 2, 1, conv_config);
        let to_out = nn::conv2d(vs / "to_out", inner_dim, dim, 1, Default::default());

        Self {
            norm,
            to_q,
            to_kv,
            to_out,
            heads,
            dim_head,
            patch_size,
            scale: (dim_head as f64).powf(-0.5),
            dropout,
        }
    }
}

impl nn::ModuleT for LocalAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs.apply(&self.norm);
        let size = xs.size();
        let batch = size[0];
        let height = size[2];
        let width = size[3];
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let windows = window_partition(&xs, self.patch_size, self.patch_size);
        let q = windows.apply(&self.to_q);
        let kv = windows.apply(&self.to_kv).chunk(2, 1);
        let window_count = q.size()[0];
        let tokens = self.patch_size * self.patch_size;

        let q = q
            .view([
                window_count,
                self.heads,
                self.dim_head,
                self.patch_size,
                self.patch_size,
            ])
            .permute([0, 1, 3, 4, 2])
            .contiguous()
            .view([window_count * self.heads, tokens, self.dim_head]);
        let k = kv[0]
            .view([
                window_count,
                self.heads,
                self.dim_head,
                self.patch_size,
                self.patch_size,
            ])
            .permute([0, 1, 3, 4, 2])
            .contiguous()
            .view([window_count * self.heads, tokens, self.dim_head]);
        let v = kv[1]
            .view([
                window_count,
                self.heads,
                self.dim_head,
                self.patch_size,
                self.patch_size,
            ])
            .permute([0, 1, 3, 4, 2])
            .contiguous()
            .view([window_count * self.heads, tokens, self.dim_head]);

        let attn = (q.matmul(&k.transpose(-1, -2)) * self.scale)
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train);
        let out = attn.matmul(&v);
        let out = out
            .view([
                window_count,
                self.heads,
                self.patch_size,
                self.patch_size,
                self.dim_head,
            ])
            .permute([0, 1, 4, 2, 3])
            .contiguous()
            .view([
                window_count,
                self.heads * self.dim_head,
                self.patch_size,
                self.patch_size,
            ])
            .apply(&self.to_out)
            .dropout(self.dropout, train);

        let _ = (grid_h, grid_w);

        window_unpartition(&out, batch, height, width, self.patch_size, self.patch_size)
    }
}

#[derive(Debug)]
struct GlobalAttention {
    norm: ChannelLayerNorm2D,
    to_q: nn::Conv2D,
    to_kv: nn::Conv2D,
    to_out: nn::Conv2D,
    heads: i64,
    dim_head: i64,
    scale: f64,
    dropout: f64,
}

impl GlobalAttention {
    fn new(
        vs: &nn::Path,
        dim: i64,
        heads: i64,
        dim_head: i64,
        global_k: i64,
        dropout: f64,
    ) -> Self {
        assert!(global_k > 0, "global k must be positive");
        assert!(heads > 0, "heads must be positive");
        assert!(dim_head > 0, "dim head must be positive");
        let inner_dim = heads * dim_head;
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), dim, 1e-5);
        let q_config = nn::ConvConfig {
            bias: false,
            ..Default::default()
        };
        let kv_config = nn::ConvConfig {
            stride: global_k,
            bias: false,
            ..Default::default()
        };
        let to_q = nn::conv2d(vs / "to_q", dim, inner_dim, 1, q_config);
        let to_kv = nn::conv2d(vs / "to_kv", dim, inner_dim * 2, global_k, kv_config);
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

impl nn::ModuleT for GlobalAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs.apply(&self.norm);
        let q = xs.apply(&self.to_q);
        let kv = xs.apply(&self.to_kv).chunk(2, 1);
        let size = q.size();
        let batch = size[0];
        let height = size[2];
        let width = size[3];
        let q_tokens = height * width;
        let kv_height = kv[0].size()[2];
        let kv_width = kv[0].size()[3];
        let kv_tokens = kv_height * kv_width;

        let q = q
            .view([batch, self.heads, self.dim_head, height, width])
            .permute([0, 1, 3, 4, 2])
            .contiguous()
            .view([batch * self.heads, q_tokens, self.dim_head]);
        let k = kv[0]
            .view([batch, self.heads, self.dim_head, kv_height, kv_width])
            .permute([0, 1, 3, 4, 2])
            .contiguous()
            .view([batch * self.heads, kv_tokens, self.dim_head]);
        let v = kv[1]
            .view([batch, self.heads, self.dim_head, kv_height, kv_width])
            .permute([0, 1, 3, 4, 2])
            .contiguous()
            .view([batch * self.heads, kv_tokens, self.dim_head]);

        let attn = (q.matmul(&k.transpose(-1, -2)) * self.scale)
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train);

        attn.matmul(&v)
            .view([batch, self.heads, height, width, self.dim_head])
            .permute([0, 1, 4, 2, 3])
            .contiguous()
            .view([batch, self.heads * self.dim_head, height, width])
            .apply(&self.to_out)
            .dropout(self.dropout, train)
    }
}

#[derive(Debug)]
struct TwinsTransformerLayer {
    local_attention: Option<LocalAttention>,
    local_ff: Option<FeedForward2D>,
    global_attention: GlobalAttention,
    global_ff: FeedForward2D,
}

impl TwinsTransformerLayer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        config: TwinsSVTStageConfig,
        heads: i64,
        dim_head: i64,
        mlp_mult: i64,
        dropout: f64,
        has_local: bool,
    ) -> Self {
        let local_attention = has_local.then(|| {
            LocalAttention::new(
                &(vs / "local_attention"),
                dim,
                heads,
                dim_head,
                config.local_patch_size,
                dropout,
            )
        });
        let local_ff =
            has_local.then(|| FeedForward2D::new(&(vs / "local_ff"), dim, mlp_mult, dropout));
        let global_attention = GlobalAttention::new(
            &(vs / "global_attention"),
            dim,
            heads,
            dim_head,
            config.global_k,
            dropout,
        );
        let global_ff = FeedForward2D::new(&(vs / "global_ff"), dim, mlp_mult, dropout);

        Self {
            local_attention,
            local_ff,
            global_attention,
            global_ff,
        }
    }
}

impl nn::ModuleT for TwinsTransformerLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        if let Some(local_attention) = &self.local_attention {
            xs = &xs + local_attention.forward_t(&xs, train);
        }

        if let Some(local_ff) = &self.local_ff {
            xs = &xs + local_ff.forward_t(&xs, train);
        }

        xs = &xs + self.global_attention.forward_t(&xs, train);
        let ff = self.global_ff.forward_t(&xs, train);

        xs + ff
    }
}

#[derive(Debug)]
struct TwinsTransformer {
    layers: Vec<TwinsTransformerLayer>,
}

impl TwinsTransformer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        depth: usize,
        config: TwinsSVTStageConfig,
        heads: i64,
        dim_head: i64,
        mlp_mult: i64,
        dropout: f64,
        has_local: bool,
    ) -> Self {
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                TwinsTransformerLayer::new(
                    &(vs / name.as_str()),
                    dim,
                    config,
                    heads,
                    dim_head,
                    mlp_mult,
                    dropout,
                    has_local,
                )
            })
            .collect();

        Self { layers }
    }
}

impl nn::ModuleT for TwinsTransformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs
    }
}

#[derive(Debug)]
struct PEG {
    proj: nn::Conv2D,
}

impl PEG {
    fn new(vs: &nn::Path, dim: i64, kernel_size: i64) -> Self {
        assert!(kernel_size > 0, "peg kernel size must be positive");
        let proj = nn::conv2d(
            vs / "proj",
            dim,
            dim,
            kernel_size,
            nn::ConvConfig {
                padding: kernel_size / 2,
                groups: dim,
                ..Default::default()
            },
        );

        Self { proj }
    }
}

impl nn::Module for PEG {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs + xs.apply(&self.proj)
    }
}

#[derive(Debug)]
struct TwinsStage {
    patch_embedding: PatchEmbedding,
    pre_peg: TwinsTransformer,
    peg: PEG,
    transformer: TwinsTransformer,
}

impl TwinsStage {
    fn new(
        vs: &nn::Path,
        dim_in: i64,
        config: TwinsSVTStageConfig,
        shared: &TwinsSVTConfig,
        is_last: bool,
    ) -> Self {
        let has_local = !is_last;
        let patch_embedding = PatchEmbedding::new(
            &(vs / "patch_embedding"),
            dim_in,
            config.emb_dim,
            config.patch_size,
        );
        let pre_peg = TwinsTransformer::new(
            &(vs / "pre_peg"),
            config.emb_dim,
            1,
            config,
            shared.heads,
            shared.dim_head,
            shared.mlp_mult,
            shared.dropout,
            has_local,
        );
        let peg = PEG::new(&(vs / "peg"), config.emb_dim, shared.peg_kernel_size);
        let transformer = TwinsTransformer::new(
            &(vs / "transformer"),
            config.emb_dim,
            config.depth,
            config,
            shared.heads,
            shared.dim_head,
            shared.mlp_mult,
            shared.dropout,
            has_local,
        );

        Self {
            patch_embedding,
            pre_peg,
            peg,
            transformer,
        }
    }
}

impl nn::ModuleT for TwinsStage {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = self.patch_embedding.forward_t(xs, train);
        let xs = self.pre_peg.forward_t(&xs, train);
        let xs = xs.apply(&self.peg);

        self.transformer.forward_t(&xs, train)
    }
}

#[derive(Debug)]
pub struct TwinsSVT {
    stages: Vec<TwinsStage>,
    head: nn::Linear,
}

impl TwinsSVT {
    pub fn new(vs: &nn::Path, config: TwinsSVTConfig) -> Self {
        let mut dim = config.channels;
        let mut stages = Vec::with_capacity(config.stages.len());

        for (index, stage_config) in config.stages.iter().copied().enumerate() {
            let name = format!("stage_{index}");
            let is_last = index == config.stages.len() - 1;
            stages.push(TwinsStage::new(
                &(vs / name.as_str()),
                dim,
                stage_config,
                &config,
                is_last,
            ));
            dim = stage_config.emb_dim;
        }

        let head = nn::linear(vs / "head", dim, config.num_classes, Default::default());

        Self { stages, head }
    }
}

impl nn::ModuleT for TwinsSVT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for stage in &self.stages {
            xs = stage.forward_t(&xs, train);
        }

        xs.adaptive_avg_pool2d([1, 1]).flat_view().apply(&self.head)
    }
}
