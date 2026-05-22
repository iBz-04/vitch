use tch::{Tensor, nn};

use crate::{config::TwinsSVTConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct TwinsSVT {
    model: CompactImageTransformer,
}

impl TwinsSVT {
    pub fn new(vs: &nn::Path, config: TwinsSVTConfig) -> Self {
        Self {
            model: CompactImageTransformer::new(&(vs / "model"), config),
        }
    }
}

impl nn::ModuleT for TwinsSVT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.model.forward_t(xs, train)
    }
}
