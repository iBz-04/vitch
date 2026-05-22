use tch::{Kind, Tensor, nn};

#[derive(Debug, Clone, Copy)]
pub struct PatchDropout {
    prob: f64,
}

impl PatchDropout {
    pub fn new(prob: f64) -> Self {
        assert!((0.0..1.0).contains(&prob));
        Self { prob }
    }
}

impl nn::ModuleT for PatchDropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        if !train || self.prob == 0.0 {
            return xs.shallow_clone();
        }

        let size = xs.size();
        assert_eq!(
            size.len(),
            3,
            "expected token tensor with shape [batch, tokens, dim]"
        );
        let batch = size[0];
        let tokens = size[1];
        let dim = size[2];
        let keep = 1.max((tokens as f64 * (1.0 - self.prob)) as i64);
        let indices = Tensor::rand([batch, tokens], (Kind::Float, xs.device()))
            .topk(keep, -1, true, false)
            .1
            .unsqueeze(-1)
            .repeat([1, 1, dim]);

        xs.gather(1, &indices, false)
    }
}
