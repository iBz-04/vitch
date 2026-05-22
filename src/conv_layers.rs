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

#[derive(Debug)]
pub struct ConvBnSiLU {
    conv: nn::Conv2D,
    bn: nn::BatchNorm,
}

impl ConvBnSiLU {
    pub fn new(
        vs: &nn::Path,
        dim_in: i64,
        dim_out: i64,
        kernel_size: i64,
        stride: i64,
        padding: i64,
        groups: i64,
    ) -> Self {
        assert!(kernel_size > 0, "kernel size must be positive");
        assert!(stride > 0, "stride must be positive");
        assert!(groups > 0, "groups must be positive");
        let conv = nn::conv2d(
            vs / "conv",
            dim_in,
            dim_out,
            kernel_size,
            nn::ConvConfig {
                stride,
                padding,
                groups,
                bias: false,
                ..Default::default()
            },
        );
        let bn = nn::batch_norm2d(vs / "bn", dim_out, Default::default());

        Self { conv, bn }
    }
}

impl nn::ModuleT for ConvBnSiLU {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.conv).apply_t(&self.bn, train).silu()
    }
}

#[derive(Debug)]
pub struct SqueezeExcite {
    fc1: nn::Conv2D,
    fc2: nn::Conv2D,
}

impl SqueezeExcite {
    pub fn new(vs: &nn::Path, dim: i64, hidden_dim: i64) -> Self {
        assert!(hidden_dim > 0, "hidden dimension must be positive");
        let fc1 = nn::conv2d(vs / "fc1", dim, hidden_dim, 1, Default::default());
        let fc2 = nn::conv2d(vs / "fc2", hidden_dim, dim, 1, Default::default());

        Self { fc1, fc2 }
    }
}

impl nn::Module for SqueezeExcite {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let gate = xs
            .adaptive_avg_pool2d([1, 1])
            .apply(&self.fc1)
            .silu()
            .apply(&self.fc2)
            .sigmoid();

        xs * gate
    }
}

#[derive(Debug)]
pub struct MbConv {
    expand: Option<ConvBnSiLU>,
    depthwise: ConvBnSiLU,
    squeeze_excite: SqueezeExcite,
    project_conv: nn::Conv2D,
    project_bn: nn::BatchNorm,
    residual: bool,
    drop_prob: f64,
}

impl MbConv {
    pub fn new(
        vs: &nn::Path,
        dim_in: i64,
        dim_out: i64,
        stride: i64,
        expansion: i64,
        shrinkage: i64,
        drop_prob: f64,
    ) -> Self {
        assert!(expansion > 0, "expansion must be positive");
        assert!(shrinkage > 0, "shrinkage must be positive");
        let hidden_dim = dim_in * expansion;
        let expand = if expansion == 1 {
            None
        } else {
            Some(ConvBnSiLU::new(
                &(vs / "expand"),
                dim_in,
                hidden_dim,
                1,
                1,
                0,
                1,
            ))
        };
        let depthwise = ConvBnSiLU::new(
            &(vs / "depthwise"),
            hidden_dim,
            hidden_dim,
            3,
            stride,
            1,
            hidden_dim,
        );
        let squeeze_excite = SqueezeExcite::new(
            &(vs / "squeeze_excite"),
            hidden_dim,
            1.max(dim_in / shrinkage),
        );
        let project_conv = nn::conv2d(
            vs / "project_conv",
            hidden_dim,
            dim_out,
            1,
            nn::ConvConfig {
                bias: false,
                ..Default::default()
            },
        );
        let project_bn = nn::batch_norm2d(vs / "project_bn", dim_out, Default::default());
        let residual = stride == 1 && dim_in == dim_out;

        Self {
            expand,
            depthwise,
            squeeze_excite,
            project_conv,
            project_bn,
            residual,
            drop_prob,
        }
    }
}

impl nn::ModuleT for MbConv {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut out = match &self.expand {
            Some(expand) => expand.forward_t(xs, train),
            None => xs.shallow_clone(),
        };
        out = self.depthwise.forward_t(&out, train);
        out = out.apply(&self.squeeze_excite);
        out = out
            .apply(&self.project_conv)
            .apply_t(&self.project_bn, train);
        out = crate::layers::drop_path(&out, self.drop_prob, train);

        if self.residual { xs + out } else { out }
    }
}
