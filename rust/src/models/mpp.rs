use tch::{Reduction, Tensor, nn, nn::ModuleT};

use crate::{config::MPPConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct MPP {
    encoder: CompactImageTransformer,
    reconstruction: nn::Linear,
}

impl MPP {
    pub fn new(vs: &nn::Path, config: MPPConfig) -> Self {
        let reconstruction = nn::linear(
            vs / "reconstruction",
            config.num_classes,
            config.num_classes,
            Default::default(),
        );

        Self {
            encoder: CompactImageTransformer::new(&(vs / "encoder"), config),
            reconstruction,
        }
    }

    pub fn forward_t_with_target(&self, xs: &Tensor, target: &Tensor, train: bool) -> Tensor {
        self.forward_t(xs, train)
            .apply(&self.reconstruction)
            .mse_loss(target, Reduction::Mean)
    }
}

impl nn::ModuleT for MPP {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.encoder.forward_t(xs, train)
    }
}
