use tch::{IndexOp, Kind, Tensor, nn};

use crate::{
    config::CCTConfig, layers::Transformer, positional::posemb_sincos_1d, tensor::repeat_token,
};

#[derive(Debug)]
struct Tokenizer {
    conv: nn::Conv2D,
    kernel_size: i64,
    stride: i64,
    padding: i64,
    pooling_kernel_size: i64,
    pooling_stride: i64,
    pooling_padding: i64,
}

impl Tokenizer {
    fn new(vs: &nn::Path, config: &CCTConfig) -> Self {
        assert!(config.kernel_size > 0, "kernel size must be positive");
        assert!(config.stride > 0, "stride must be positive");
        assert!(
            config.pooling_kernel_size > 0,
            "pooling kernel size must be positive"
        );
        assert!(config.pooling_stride > 0, "pooling stride must be positive");
        let conv = nn::conv2d(
            vs / "conv",
            config.channels,
            config.embedding_dim,
            config.kernel_size,
            nn::ConvConfig {
                stride: config.stride,
                padding: config.padding,
                bias: false,
                ..Default::default()
            },
        );

        Self {
            conv,
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            pooling_kernel_size: config.pooling_kernel_size,
            pooling_stride: config.pooling_stride,
            pooling_padding: config.pooling_padding,
        }
    }

    fn sequence_length(&self, image_height: i64, image_width: i64) -> i64 {
        let conv_height =
            conv_output_size(image_height, self.kernel_size, self.padding, self.stride);
        let conv_width = conv_output_size(image_width, self.kernel_size, self.padding, self.stride);
        let pool_height = conv_output_size(
            conv_height,
            self.pooling_kernel_size,
            self.pooling_padding,
            self.pooling_stride,
        );
        let pool_width = conv_output_size(
            conv_width,
            self.pooling_kernel_size,
            self.pooling_padding,
            self.pooling_stride,
        );

        pool_height * pool_width
    }
}

impl nn::ModuleT for Tokenizer {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        let xs = xs.apply(&self.conv).relu().max_pool2d(
            [self.pooling_kernel_size, self.pooling_kernel_size],
            [self.pooling_stride, self.pooling_stride],
            [self.pooling_padding, self.pooling_padding],
            [1, 1],
            false,
        );
        let size = xs.size();
        let batch = size[0];
        let channels = size[1];
        let height = size[2];
        let width = size[3];

        xs.view([batch, channels, height * width]).transpose(1, 2)
    }
}

#[derive(Debug)]
pub struct CCT {
    tokenizer: Tokenizer,
    cls_token: Option<Tensor>,
    pos_embedding: Tensor,
    transformer: Transformer,
    attention_pool: Option<nn::Linear>,
    norm: nn::LayerNorm,
    head: nn::Linear,
    seq_pool: bool,
    dropout: f64,
}

impl CCT {
    pub fn new(vs: &nn::Path, config: CCTConfig) -> Self {
        assert!(config.embedding_dim > 0, "embedding dim must be positive");
        assert!(config.num_heads > 0, "num heads must be positive");
        assert_eq!(
            config.embedding_dim % config.num_heads,
            0,
            "embedding dim must be divisible by num heads"
        );
        let tokenizer = Tokenizer::new(&(vs / "tokenizer"), &config);
        let sequence_length =
            tokenizer.sequence_length(config.image_size.height, config.image_size.width);
        assert!(sequence_length > 0, "tokenizer produced no tokens");
        let pos_tokens = if config.seq_pool {
            sequence_length
        } else {
            sequence_length + 1
        };
        let cls_token = if config.seq_pool {
            None
        } else {
            Some(vs.var(
                "cls_token",
                &[1, 1, config.embedding_dim],
                nn::Init::Randn {
                    mean: 0.0,
                    stdev: 1.0,
                },
            ))
        };
        let pos_embedding =
            posemb_sincos_1d(pos_tokens, config.embedding_dim, 10000.0, vs.device()).unsqueeze(0);
        let transformer = Transformer::new(
            &(vs / "transformer"),
            config.embedding_dim,
            config.num_layers,
            config.num_heads,
            config.embedding_dim / config.num_heads,
            config.embedding_dim * config.mlp_ratio,
            config.dropout,
        );
        let attention_pool = if config.seq_pool {
            Some(nn::linear(
                vs / "attention_pool",
                config.embedding_dim,
                1,
                Default::default(),
            ))
        } else {
            None
        };
        let norm = nn::layer_norm(vs / "norm", vec![config.embedding_dim], Default::default());
        let head = nn::linear(
            vs / "head",
            config.embedding_dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            tokenizer,
            cls_token,
            pos_embedding,
            transformer,
            attention_pool,
            norm,
            head,
            seq_pool: config.seq_pool,
            dropout: config.dropout,
        }
    }
}

impl nn::ModuleT for CCT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = self.tokenizer.forward_t(xs, train);
        let batch = xs.size()[0];

        if let Some(cls_token) = &self.cls_token {
            xs = Tensor::cat(&[repeat_token(cls_token, batch), xs], 1);
        }

        let tokens = xs.size()[1];
        xs = (xs + self.pos_embedding.i((.., 0..tokens, ..))).dropout(self.dropout, train);
        let xs = self.transformer.forward_t(&xs, train).apply(&self.norm);
        let pooled = if self.seq_pool {
            let attention_pool = self
                .attention_pool
                .as_ref()
                .expect("sequence pooling head is required");
            let weights = xs
                .apply(attention_pool)
                .squeeze_dim(-1)
                .softmax(1, Kind::Float);

            (weights.unsqueeze(-1) * xs).sum_dim_intlist(&[1_i64][..], false, Kind::Float)
        } else {
            xs.i((.., 0))
        };

        pooled.apply(&self.head)
    }
}

fn conv_output_size(size: i64, kernel_size: i64, padding: i64, stride: i64) -> i64 {
    ((size + 2 * padding - kernel_size) / stride) + 1
}
