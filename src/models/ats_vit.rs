use tch::{Kind, Tensor, nn};

use crate::{config::ATSConfig, models::compact::MaskedImageTransformer};

#[derive(Debug)]
pub struct ATS {
    model: MaskedImageTransformer,
    keep_ratio: f64,
}

impl ATS {
    pub fn new(vs: &nn::Path, config: ATSConfig) -> Self {
        assert!(
            config.keep_ratio > 0.0 && config.keep_ratio <= 1.0,
            "keep ratio must be in (0, 1]"
        );
        let keep_ratio = config.keep_ratio;
        Self {
            model: MaskedImageTransformer::new(&(vs / "model"), config.vision),
            keep_ratio,
        }
    }

    pub fn forward_t_with_keep_ratio(&self, xs: &Tensor, keep_ratio: f64, train: bool) -> Tensor {
        assert!(
            keep_ratio > 0.0 && keep_ratio <= 1.0,
            "keep ratio must be in (0, 1]"
        );
        let batch = xs.size()[0];
        let (grid_h, grid_w) = self.model.patch_grid_for(xs);
        let tokens = grid_h * grid_w;
        let keep = 1.max((tokens as f64 * keep_ratio) as i64);
        let scores = Tensor::rand([batch, tokens], (Kind::Float, xs.device()));
        let keep_indices = scores.argsort(-1, true).narrow(1, 0, keep);
        let mask = Tensor::zeros([batch, tokens], (Kind::Bool, xs.device())).scatter_value(
            1,
            &keep_indices,
            1,
        );

        self.model.forward_t_with_mask(xs, Some(&mask), train)
    }
}

impl nn::ModuleT for ATS {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward_t_with_keep_ratio(xs, self.keep_ratio, train)
    }
}
