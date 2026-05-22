use tch::{Kind, Tensor, nn};

pub fn l2_normalize(xs: &Tensor, dim: i64, eps: f64) -> Tensor {
    let denom = xs
        .square()
        .sum_dim_intlist(&[dim][..], true, Kind::Float)
        .clamp_min(eps)
        .sqrt();

    xs / denom
}

#[derive(Debug)]
pub struct RmsNormHeads {
    gamma: Tensor,
    scale: f64,
}

impl RmsNormHeads {
    pub fn new(vs: &nn::Path, heads: i64, dim: i64) -> Self {
        let scale = (dim as f64).sqrt();
        let gamma = vs.var("gamma", &[heads, 1, dim], nn::Init::Const(1.0 / scale));

        Self { gamma, scale }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        l2_normalize(xs, -1, 1e-12) * self.scale * self.gamma.unsqueeze(0)
    }
}
