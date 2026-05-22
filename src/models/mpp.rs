use tch::{Reduction, Tensor, nn, nn::ModuleT};

use crate::{config::MPPConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct MPP {
    encoder: CompactImageTransformer,
    reconstruction: nn::Linear,
    reconstruction_dim: i64,
}

impl MPP {
    pub fn new(vs: &nn::Path, config: MPPConfig) -> Self {
        let encoder_dim = config.encoder.num_classes;
        let encoder = CompactImageTransformer::new(&(vs / "encoder"), config.encoder);
        let reconstruction = nn::linear(
            vs / "reconstruction",
            encoder_dim,
            config.reconstruction_dim,
            Default::default(),
        );

        Self {
            encoder,
            reconstruction,
            reconstruction_dim: config.reconstruction_dim,
        }
    }

    pub fn forward_t_with_target(&self, xs: &Tensor, target: &Tensor, train: bool) -> Tensor {
        let reconstruction = self.forward_t(xs, train).apply(&self.reconstruction);
        assert_eq!(
            target.size()[1],
            self.reconstruction_dim,
            "target dim must match reconstruction dim"
        );

        reconstruction.mse_loss(target, Reduction::Mean)
    }
}

impl nn::ModuleT for MPP {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.encoder.forward_t(xs, train)
    }
}
