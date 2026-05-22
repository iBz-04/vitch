use tch::{Tensor, nn};

use crate::{config::CrossFormerConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct CrossFormer {
    model: CompactImageTransformer,
}

impl CrossFormer {
    pub fn new(vs: &nn::Path, config: CrossFormerConfig) -> Self {
        Self {
            model: CompactImageTransformer::new(&(vs / "model"), config),
        }
    }
}

impl nn::ModuleT for CrossFormer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.model.forward_t(xs, train)
    }
}
