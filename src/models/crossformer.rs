use tch::{Kind, Tensor, nn};

use crate::{
    config::CrossFormerConfig,
    conv_layers::ChannelLayerNorm2D,
    tensor::{assert_image_patchable, window_partition, window_unpartition},
};

fn require_stage_len<T>(values: &[T], stages: usize, name: &str) {
    assert_eq!(values.len(), stages, "{name} must match dim length");
}

#[derive(Debug)]
struct CrossEmbedLayer {
    convs: Vec<nn::Conv2D>,
}

impl CrossEmbedLayer {
    fn new(vs: &nn::Path, dim_in: i64, dim_out: i64, kernel_sizes: &[i64], stride: i64) -> Self {
        assert!(!kernel_sizes.is_empty(), "kernel sizes must not be empty");
        assert!(stride > 0, "stride must be positive");
        let mut kernel_sizes = kernel_sizes.to_vec();
        kernel_sizes.sort_unstable();
        let num_scales = kernel_sizes.len();
        let mut dim_scales = (1..num_scales)
            .map(|index| dim_out / (1_i64 << index))
            .collect::<Vec<_>>();
        dim_scales.push(dim_out - dim_scales.iter().sum::<i64>());
        let convs = kernel_sizes
            .iter()
            .zip(dim_scales)
            .enumerate()
            .map(|(index, (&kernel_size, dim_scale))| {
                assert!(kernel_size > 0, "kernel size must be positive");
                let name = format!("conv_{index}");
                nn::conv2d(
                    vs / name.as_str(),
                    dim_in,
                    dim_scale,
                    kernel_size,
                    nn::ConvConfig {
                        stride,
                        padding: (kernel_size - stride) / 2,
                        ..Default::default()
                    },
                )
            })
            .collect();

        Self { convs }
    }
}

impl nn::Module for CrossEmbedLayer {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let maps = self
            .convs
            .iter()
            .map(|conv| xs.apply(conv))
            .collect::<Vec<_>>();
        let refs = maps.iter().collect::<Vec<_>>();

        Tensor::cat(&refs, 1)
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
    fn new(vs: &nn::Path, dim: i64, dropout: f64) -> Self {
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), dim, 1e-5);
        let fc1 = nn::conv2d(vs / "fc1", dim, dim * 4, 1, Default::default());
        let fc2 = nn::conv2d(vs / "fc2", dim * 4, dim, 1, Default::default());

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
    }
}

#[derive(Debug, Clone, Copy)]
enum AttentionType {
    Short,
    Long,
}

#[derive(Debug)]
struct Attention {
    norm: ChannelLayerNorm2D,
    to_qkv: nn::Conv2D,
    to_out: nn::Conv2D,
    heads: i64,
    dim_head: i64,
    scale: f64,
    window_size: i64,
    attn_type: AttentionType,
    dropout: f64,
}

impl Attention {
    fn new(
        vs: &nn::Path,
        dim: i64,
        attn_type: AttentionType,
        window_size: i64,
        dim_head: i64,
        dropout: f64,
    ) -> Self {
        assert!(dim_head > 0, "dim head must be positive");
        assert!(window_size > 0, "window size must be positive");
        assert_eq!(dim % dim_head, 0, "dim must be divisible by dim head");
        let heads = dim / dim_head;
        let inner_dim = heads * dim_head;
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), dim, 1e-5);
        let to_qkv = nn::conv2d(
            vs / "to_qkv",
            dim,
            inner_dim * 3,
            1,
            nn::ConvConfig {
                bias: false,
                ..Default::default()
            },
        );
        let to_out = nn::conv2d(vs / "to_out", inner_dim, dim, 1, Default::default());

        Self {
            norm,
            to_qkv,
            to_out,
            heads,
            dim_head,
            scale: (dim_head as f64).powf(-0.5),
            window_size,
            attn_type,
            dropout,
        }
    }
}

impl nn::ModuleT for Attention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        let batch = size[0];
        let channels = size[1];
        let height = size[2];
        let width = size[3];
        assert_image_patchable(height, width, self.window_size, self.window_size);
        let grid_h = height / self.window_size;
        let grid_w = width / self.window_size;
        let windows = match self.attn_type {
            AttentionType::Short => {
                window_partition(&xs.apply(&self.norm), self.window_size, self.window_size)
            }
            AttentionType::Long => xs
                .apply(&self.norm)
                .view([
                    batch,
                    channels,
                    self.window_size,
                    grid_h,
                    self.window_size,
                    grid_w,
                ])
                .permute([0, 3, 5, 1, 2, 4])
                .contiguous()
                .view([
                    batch * grid_h * grid_w,
                    channels,
                    self.window_size,
                    self.window_size,
                ]),
        };
        let qkv = windows.apply(&self.to_qkv).chunk(3, 1);
        let window_count = qkv[0].size()[0];
        let tokens = self.window_size * self.window_size;
        let q = qkv[0]
            .view([
                window_count,
                self.heads,
                self.dim_head,
                self.window_size,
                self.window_size,
            ])
            .permute([0, 1, 3, 4, 2])
            .contiguous()
            .view([window_count, self.heads, tokens, self.dim_head]);
        let k = qkv[1]
            .view([
                window_count,
                self.heads,
                self.dim_head,
                self.window_size,
                self.window_size,
            ])
            .permute([0, 1, 3, 4, 2])
            .contiguous()
            .view([window_count, self.heads, tokens, self.dim_head]);
        let v = qkv[2]
            .view([
                window_count,
                self.heads,
                self.dim_head,
                self.window_size,
                self.window_size,
            ])
            .permute([0, 1, 3, 4, 2])
            .contiguous()
            .view([window_count, self.heads, tokens, self.dim_head]);
        let attn = ((&q * self.scale).matmul(&k.transpose(-1, -2)))
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train);
        let out = attn
            .matmul(&v)
            .view([
                window_count,
                self.heads,
                self.window_size,
                self.window_size,
                self.dim_head,
            ])
            .permute([0, 1, 4, 2, 3])
            .contiguous()
            .view([
                window_count,
                self.heads * self.dim_head,
                self.window_size,
                self.window_size,
            ])
            .apply(&self.to_out);

        match self.attn_type {
            AttentionType::Short => window_unpartition(
                &out,
                batch,
                height,
                width,
                self.window_size,
                self.window_size,
            ),
            AttentionType::Long => out
                .view([
                    batch,
                    grid_h,
                    grid_w,
                    self.heads * self.dim_head,
                    self.window_size,
                    self.window_size,
                ])
                .permute([0, 3, 4, 1, 5, 2])
                .contiguous()
                .view([batch, self.heads * self.dim_head, height, width]),
        }
    }
}

#[derive(Debug)]
struct TransformerLayer {
    short_attention: Attention,
    short_ff: FeedForward2D,
    long_attention: Attention,
    long_ff: FeedForward2D,
}

impl TransformerLayer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        local_window_size: i64,
        global_window_size: i64,
        dim_head: i64,
        attn_dropout: f64,
        ff_dropout: f64,
    ) -> Self {
        let short_attention = Attention::new(
            &(vs / "short_attention"),
            dim,
            AttentionType::Short,
            local_window_size,
            dim_head,
            attn_dropout,
        );
        let short_ff = FeedForward2D::new(&(vs / "short_ff"), dim, ff_dropout);
        let long_attention = Attention::new(
            &(vs / "long_attention"),
            dim,
            AttentionType::Long,
            global_window_size,
            dim_head,
            attn_dropout,
        );
        let long_ff = FeedForward2D::new(&(vs / "long_ff"), dim, ff_dropout);

        Self {
            short_attention,
            short_ff,
            long_attention,
            long_ff,
        }
    }
}

impl nn::ModuleT for TransformerLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + self.short_attention.forward_t(xs, train);
        let xs = &xs + self.short_ff.forward_t(&xs, train);
        let xs = &xs + self.long_attention.forward_t(&xs, train);
        let ff = self.long_ff.forward_t(&xs, train);

        xs + ff
    }
}

#[derive(Debug)]
struct Transformer {
    layers: Vec<TransformerLayer>,
}

impl Transformer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        local_window_size: i64,
        global_window_size: i64,
        depth: usize,
        config: &CrossFormerConfig,
    ) -> Self {
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                TransformerLayer::new(
                    &(vs / name.as_str()),
                    dim,
                    local_window_size,
                    global_window_size,
                    config.dim_head,
                    config.attn_dropout,
                    config.ff_dropout,
                )
            })
            .collect();

        Self { layers }
    }
}

impl nn::ModuleT for Transformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        xs
    }
}

#[derive(Debug)]
struct CrossFormerStage {
    cross_embed: CrossEmbedLayer,
    transformer: Transformer,
}

impl CrossFormerStage {
    fn new(
        vs: &nn::Path,
        dim_in: i64,
        dim: i64,
        depth: usize,
        global_window_size: i64,
        local_window_size: i64,
        kernel_sizes: &[i64],
        stride: i64,
        config: &CrossFormerConfig,
    ) -> Self {
        let cross_embed =
            CrossEmbedLayer::new(&(vs / "cross_embed"), dim_in, dim, kernel_sizes, stride);
        let transformer = Transformer::new(
            &(vs / "transformer"),
            dim,
            local_window_size,
            global_window_size,
            depth,
            config,
        );

        Self {
            cross_embed,
            transformer,
        }
    }
}

impl nn::ModuleT for CrossFormerStage {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs.apply(&self.cross_embed);

        self.transformer.forward_t(&xs, train)
    }
}

#[derive(Debug)]
pub struct CrossFormer {
    stages: Vec<CrossFormerStage>,
    head: nn::Linear,
}

impl CrossFormer {
    pub fn new(vs: &nn::Path, config: CrossFormerConfig) -> Self {
        assert!(!config.dim.is_empty(), "dim must not be empty");
        let stages_len = config.dim.len();
        require_stage_len(&config.depth, stages_len, "depth");
        require_stage_len(&config.global_window_size, stages_len, "global window size");
        require_stage_len(&config.local_window_size, stages_len, "local window size");
        require_stage_len(
            &config.cross_embed_kernel_sizes,
            stages_len,
            "cross embed kernel sizes",
        );
        require_stage_len(
            &config.cross_embed_strides,
            stages_len,
            "cross embed strides",
        );
        let mut dim_in = config.channels;
        let mut stages = Vec::with_capacity(stages_len);

        for index in 0..stages_len {
            let name = format!("stage_{index}");
            stages.push(CrossFormerStage::new(
                &(vs / name.as_str()),
                dim_in,
                config.dim[index],
                config.depth[index],
                config.global_window_size[index],
                config.local_window_size[index],
                &config.cross_embed_kernel_sizes[index],
                config.cross_embed_strides[index],
                &config,
            ));
            dim_in = config.dim[index];
        }

        let last_dim = *config.dim.last().expect("dim must not be empty");
        let head = nn::linear(
            vs / "head",
            last_dim,
            config.num_classes,
            Default::default(),
        );

        Self { stages, head }
    }
}

impl nn::ModuleT for CrossFormer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for stage in &self.stages {
            xs = stage.forward_t(&xs, train);
        }

        xs.mean_dim(&[2_i64, 3][..], false, Kind::Float)
            .apply(&self.head)
    }
}
