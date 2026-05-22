use tch::{Kind, Tensor, nn};

#[derive(Debug)]
pub struct ChannelLayerNorm2D {
    gamma: Tensor,
    beta: Tensor,
    eps: f64,
}

impl ChannelLayerNorm2D {
    pub fn new(vs: &nn::Path, dim: i64, eps: f64) -> Self {
        let gamma = vs.ones("gamma", &[1, dim, 1, 1]);
        let beta = vs.zeros("beta", &[1, dim, 1, 1]);

        Self { gamma, beta, eps }
    }
}

impl nn::Module for ChannelLayerNorm2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mean = xs.mean_dim(1, true, Kind::Float);
        let var = xs.var_dim(&[1_i64][..], false, true);

        (xs - mean) / (var + self.eps).sqrt() * &self.gamma + &self.beta
    }
}

#[derive(Debug)]
pub struct DepthwiseConv2D {
    depthwise: nn::Conv2D,
    pointwise: nn::Conv2D,
}

impl DepthwiseConv2D {
    pub fn new(
        vs: &nn::Path,
        dim_in: i64,
        dim_out: i64,
        kernel_size: i64,
        padding: i64,
        stride: i64,
        bias: bool,
    ) -> Self {
        let depthwise = nn::conv2d(
            vs / "depthwise",
            dim_in,
            dim_in,
            kernel_size,
            nn::ConvConfig {
                padding,
                stride,
                groups: dim_in,
                bias,
                ..Default::default()
            },
        );
        let pointwise = nn::conv2d(
            vs / "pointwise",
            dim_in,
            dim_out,
            1,
            nn::ConvConfig {
                bias,
                ..Default::default()
            },
        );

        Self {
            depthwise,
            pointwise,
        }
    }
}

impl nn::Module for DepthwiseConv2D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.depthwise).apply(&self.pointwise)
    }
}
