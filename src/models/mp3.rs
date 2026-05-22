use tch::{Reduction, Tensor, nn, nn::ModuleT};

use crate::{config::MP3Config, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct MP3 {
    encoder: CompactImageTransformer,
    predictor: nn::Linear,
    prediction_dim: i64,
}

impl MP3 {
    pub fn new(vs: &nn::Path, config: MP3Config) -> Self {
        assert!(
            config.masking_ratio > 0.0 && config.masking_ratio < 1.0,
            "masking ratio must be in (0, 1)"
        );
        let encoder_dim = config.encoder.num_classes;
        let encoder = CompactImageTransformer::new(&(vs / "encoder"), config.encoder);
        let predictor = nn::linear(
            vs / "predictor",
            encoder_dim,
            config.prediction_dim,
            Default::default(),
        );

        Self {
            encoder,
            predictor,
            prediction_dim: config.prediction_dim,
        }
    }

    pub fn forward_t_with_target(&self, xs: &Tensor, target: &Tensor, train: bool) -> Tensor {
        let prediction = self.forward_t(xs, train).apply(&self.predictor);
        assert_eq!(
            target.size()[1],
            self.prediction_dim,
            "target dim must match prediction dim"
        );

        prediction.mse_loss(target, Reduction::Mean)
    }
}

impl nn::ModuleT for MP3 {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.encoder.forward_t(xs, train)
    }
}
