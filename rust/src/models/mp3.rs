use tch::{Reduction, Tensor, nn, nn::ModuleT};

use crate::{config::MP3Config, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct MP3 {
    encoder: CompactImageTransformer,
    predictor: nn::Linear,
}

impl MP3 {
    pub fn new(vs: &nn::Path, config: MP3Config) -> Self {
        let predictor = nn::linear(
            vs / "predictor",
            config.num_classes,
            config.num_classes,
            Default::default(),
        );

        Self {
            encoder: CompactImageTransformer::new(&(vs / "encoder"), config),
            predictor,
        }
    }

    pub fn forward_t_with_target(&self, xs: &Tensor, target: &Tensor, train: bool) -> Tensor {
        self.forward_t(xs, train)
            .apply(&self.predictor)
            .mse_loss(target, Reduction::Mean)
    }
}

impl nn::ModuleT for MP3 {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.encoder.forward_t(xs, train)
    }
}
