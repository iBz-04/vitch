use tch::{Tensor, nn};

use crate::{config::NesTConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct NesT {
    model: CompactImageTransformer,
}

impl NesT {
    pub fn new(vs: &nn::Path, config: NesTConfig) -> Self {
        Self {
            model: CompactImageTransformer::new(&(vs / "model"), config),
        }
    }
}

impl nn::ModuleT for NesT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.model.forward_t(xs, train)
    }
}
