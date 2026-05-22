use tch::{Kind, Tensor, nn, nn::Module};

use crate::{
    config::SepViTConfig,
    conv_layers::ChannelLayerNorm2D,
    tensor::{assert_image_patchable, repeat_token, window_partition, window_unpartition},
};

fn expand_stage_values(values: &[i64], stages: usize, name: &str) -> Vec<i64> {
    match values.len() {
        1 => vec![values[0]; stages],
        len if len == stages => values.to_vec(),
        _ => panic!("{name} must have length 1 or match depth length"),
    }
}

#[derive(Debug)]
struct OverlappingPatchEmbed {
    conv: nn::Conv2D,
}

impl OverlappingPatchEmbed {
    fn new(vs: &nn::Path, dim_in: i64, dim_out: i64, stride: i64) -> Self {
        assert!(stride > 0, "stride must be positive");
        let kernel_size = stride * 2 - 1;
        let conv = nn::conv2d(
            vs / "conv",
            dim_in,
            dim_out,
            kernel_size,
            nn::ConvConfig {
                stride,
                padding: kernel_size / 2,
                ..Default::default()
            },
        );

        Self { conv }
    }
}

impl nn::Module for OverlappingPatchEmbed {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.conv)
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
struct FeedForward2D {
    norm: ChannelLayerNorm2D,
    fc1: nn::Conv2D,
    fc2: nn::Conv2D,
    dropout: f64,
}

impl FeedForward2D {
    fn new(vs: &nn::Path, dim: i64, mult: i64, dropout: f64) -> Self {
        assert!(mult > 0, "feedforward multiplier must be positive");
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
struct WindowTokensToQk {
    norm: nn::LayerNorm,
    conv: nn::Conv1D,
    heads: i64,
    dim_head: i64,
}

impl WindowTokensToQk {
    fn new(vs: &nn::Path, heads: i64, dim_head: i64) -> Self {
        let inner_dim = heads * dim_head;
        let norm = nn::layer_norm(vs / "norm", vec![dim_head], Default::default());
        let conv = nn::conv1d(vs / "conv", inner_dim, inner_dim * 2, 1, Default::default());

        Self {
            norm,
            conv,
            heads,
            dim_head,
        }
    }
}

impl nn::Module for WindowTokensToQk {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let size = xs.size();
        let batch = size[0];
        let tokens = size[2];

        xs.permute([0, 2, 1, 3])
            .contiguous()
            .view([batch * tokens, self.heads, self.dim_head])
            .apply(&self.norm)
            .gelu("none")
            .view([batch, tokens, self.heads, self.dim_head])
            .permute([0, 2, 3, 1])
            .contiguous()
            .view([batch, self.heads * self.dim_head, tokens])
            .apply(&self.conv)
            .view([batch, self.heads, self.dim_head * 2, tokens])
            .permute([0, 1, 3, 2])
    }
}

#[derive(Debug)]
struct DSSA {
    norm: ChannelLayerNorm2D,
    to_qkv: nn::Conv1D,
    window_tokens: Tensor,
    window_tokens_to_qk: WindowTokensToQk,
    to_out: nn::Conv2D,
    heads: i64,
    dim_head: i64,
    scale: f64,
    window_size: i64,
    dropout: f64,
}

impl DSSA {
    fn new(
        vs: &nn::Path,
        dim: i64,
        heads: i64,
        dim_head: i64,
        dropout: f64,
        window_size: i64,
    ) -> Self {
        assert!(heads > 0, "heads must be positive");
        assert!(dim_head > 0, "dim head must be positive");
        assert!(window_size > 0, "window size must be positive");
        let inner_dim = heads * dim_head;
        let norm = ChannelLayerNorm2D::new(&(vs / "norm"), dim, 1e-5);
        let to_qkv = nn::conv1d(
            vs / "to_qkv",
            dim,
            inner_dim * 3,
            1,
            nn::ConvConfig {
                bias: false,
                ..Default::default()
            },
        );
        let window_tokens = vs.var(
            "window_tokens",
            &[dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let window_tokens_to_qk =
            WindowTokensToQk::new(&(vs / "window_tokens_to_qk"), heads, dim_head);
        let to_out = nn::conv2d(vs / "to_out", inner_dim, dim, 1, Default::default());

        Self {
            norm,
            to_qkv,
            window_tokens,
            window_tokens_to_qk,
            to_out,
            heads,
            dim_head,
            scale: (dim_head as f64).powf(-0.5),
            window_size,
            dropout,
        }
    }
}

impl nn::ModuleT for DSSA {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        let batch = size[0];
        let height = size[2];
        let width = size[3];
        assert_image_patchable(height, width, self.window_size, self.window_size);

        let grid_h = height / self.window_size;
        let grid_w = width / self.window_size;
        let num_windows = grid_h * grid_w;
        let window_tokens = self.window_size * self.window_size;
        let windows = window_partition(&xs.apply(&self.norm), self.window_size, self.window_size)
            .view([batch * num_windows, size[1], window_tokens]);
        let token = repeat_token(&self.window_tokens, batch * num_windows).transpose(1, 2);
        let windows = Tensor::cat(&[token, windows], -1);
        let qkv = windows.apply(&self.to_qkv).chunk(3, 1);
        let q = qkv[0]
            .view([
                batch * num_windows,
                self.heads,
                self.dim_head,
                window_tokens + 1,
            ])
            .permute([0, 1, 3, 2]);
        let k = qkv[1]
            .view([
                batch * num_windows,
                self.heads,
                self.dim_head,
                window_tokens + 1,
            ])
            .permute([0, 1, 3, 2]);
        let v = qkv[2]
            .view([
                batch * num_windows,
                self.heads,
                self.dim_head,
                window_tokens + 1,
            ])
            .permute([0, 1, 3, 2]);
        let attn = ((&q * self.scale).matmul(&k.transpose(-1, -2)))
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train);
        let out = attn.matmul(&v);
        let window_tokens_out = out.narrow(2, 0, 1).squeeze_dim(2);
        let windowed_fmaps = out.narrow(2, 1, window_tokens);

        if num_windows == 1 {
            let fmap = windowed_fmaps
                .view(
                    &[
                        batch,
                        grid_h,
                        grid_w,
                        self.heads,
                        self.window_size,
                        self.window_size,
                        self.dim_head,
                    ][..],
                )
                .permute([0, 3, 6, 1, 4, 2, 5])
                .contiguous()
                .view([batch, self.heads * self.dim_head, height, width]);

            return fmap.apply(&self.to_out).dropout(self.dropout, train);
        }

        let window_tokens_out = window_tokens_out
            .view([batch, grid_h * grid_w, self.heads, self.dim_head])
            .permute([0, 2, 1, 3]);
        let windowed_fmaps = windowed_fmaps
            .view([
                batch,
                grid_h * grid_w,
                self.heads,
                window_tokens,
                self.dim_head,
            ])
            .permute([0, 2, 1, 3, 4]);
        let qk = self
            .window_tokens_to_qk
            .forward(&window_tokens_out)
            .chunk(2, -1);
        let window_q = &qk[0] * self.scale;
        let window_k = &qk[1];
        let window_attn = window_q
            .matmul(&window_k.transpose(-1, -2))
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train);
        let aggregated = window_attn
            .unsqueeze(-2)
            .matmul(&windowed_fmaps)
            .squeeze_dim(-2);
        let windows = aggregated
            .view([
                batch,
                self.heads,
                grid_h,
                grid_w,
                window_tokens,
                self.dim_head,
            ])
            .permute([0, 2, 3, 1, 5, 4])
            .contiguous()
            .view([
                batch * num_windows,
                self.heads * self.dim_head,
                self.window_size,
                self.window_size,
            ]);

        window_unpartition(
            &windows,
            batch,
            height,
            width,
            self.window_size,
            self.window_size,
        )
        .apply(&self.to_out)
        .dropout(self.dropout, train)
    }
}

#[derive(Debug)]
struct TransformerLayer {
    attention: DSSA,
    feed_forward: FeedForward2D,
}

impl TransformerLayer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        heads: i64,
        dim_head: i64,
        ff_mult: i64,
        dropout: f64,
        window_size: i64,
    ) -> Self {
        let attention = DSSA::new(
            &(vs / "attention"),
            dim,
            heads,
            dim_head,
            dropout,
            window_size,
        );
        let feed_forward = FeedForward2D::new(&(vs / "feed_forward"), dim, ff_mult, dropout);

        Self {
            attention,
            feed_forward,
        }
    }
}

impl nn::ModuleT for TransformerLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs + self.attention.forward_t(xs, train);
        let ff = self.feed_forward.forward_t(&xs, train);

        xs + ff
    }
}

#[derive(Debug)]
struct Transformer {
    layers: Vec<TransformerLayer>,
    norm: Option<ChannelLayerNorm2D>,
}

impl Transformer {
    fn new(
        vs: &nn::Path,
        dim: i64,
        depth: usize,
        heads: i64,
        dim_head: i64,
        ff_mult: i64,
        dropout: f64,
        window_size: i64,
        norm_output: bool,
    ) -> Self {
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                TransformerLayer::new(
                    &(vs / name.as_str()),
                    dim,
                    heads,
                    dim_head,
                    ff_mult,
                    dropout,
                    window_size,
                )
            })
            .collect();
        let norm = norm_output.then(|| ChannelLayerNorm2D::new(&(vs / "norm"), dim, 1e-5));

        Self { layers, norm }
    }
}

impl nn::ModuleT for Transformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for layer in &self.layers {
            xs = layer.forward_t(&xs, train);
        }

        match &self.norm {
            Some(norm) => xs.apply(norm),
            None => xs,
        }
    }
}

#[derive(Debug)]
struct SepViTStage {
    patch_embed: OverlappingPatchEmbed,
    peg: PEG,
    transformer: Transformer,
}

impl SepViTStage {
    fn new(
        vs: &nn::Path,
        dim_in: i64,
        dim: i64,
        depth: usize,
        stride: i64,
        heads: i64,
        window_size: i64,
        config: &SepViTConfig,
        is_last: bool,
    ) -> Self {
        let patch_embed = OverlappingPatchEmbed::new(&(vs / "patch_embed"), dim_in, dim, stride);
        let peg = PEG::new(&(vs / "peg"), dim, 3);
        let transformer = Transformer::new(
            &(vs / "transformer"),
            dim,
            depth,
            heads,
            config.dim_head,
            config.ff_mult,
            config.dropout,
            window_size,
            !is_last,
        );

        Self {
            patch_embed,
            peg,
            transformer,
        }
    }
}

impl nn::ModuleT for SepViTStage {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = xs.apply(&self.patch_embed).apply(&self.peg);

        self.transformer.forward_t(&xs, train)
    }
}

#[derive(Debug)]
pub struct SepViT {
    stages: Vec<SepViTStage>,
    norm: nn::LayerNorm,
    head: nn::Linear,
}

impl SepViT {
    pub fn new(vs: &nn::Path, config: SepViTConfig) -> Self {
        assert!(!config.depth.is_empty(), "depth must not be empty");
        assert!(config.dim > 0, "dim must be positive");
        let num_stages = config.depth.len();
        let heads = expand_stage_values(&config.heads, num_stages, "heads");
        let window_size = expand_stage_values(&config.window_size, num_stages, "window size");
        let strides = std::iter::once(4)
            .chain(std::iter::repeat_n(2, num_stages.saturating_sub(1)))
            .collect::<Vec<_>>();
        let dims = (0..num_stages)
            .map(|index| config.dim * (1_i64 << index))
            .collect::<Vec<_>>();
        let mut dim_in = config.channels;
        let mut stages = Vec::with_capacity(num_stages);

        for (index, (((&dim, &depth), &stride), (&heads, &window_size))) in dims
            .iter()
            .zip(&config.depth)
            .zip(&strides)
            .zip(heads.iter().zip(&window_size))
            .enumerate()
        {
            let name = format!("stage_{index}");
            let is_last = index == num_stages - 1;
            stages.push(SepViTStage::new(
                &(vs / name.as_str()),
                dim_in,
                dim,
                depth,
                stride,
                heads,
                window_size,
                &config,
                is_last,
            ));
            dim_in = dim;
        }

        let last_dim = *dims.last().expect("depth must not be empty");
        let norm = nn::layer_norm(vs / "norm", vec![last_dim], Default::default());
        let head = nn::linear(
            vs / "head",
            last_dim,
            config.num_classes,
            Default::default(),
        );

        Self { stages, norm, head }
    }
}

impl nn::ModuleT for SepViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = xs.shallow_clone();

        for stage in &self.stages {
            xs = stage.forward_t(&xs, train);
        }

        xs.mean_dim(&[2_i64, 3][..], false, Kind::Float)
            .apply(&self.norm)
            .apply(&self.head)
    }
}
