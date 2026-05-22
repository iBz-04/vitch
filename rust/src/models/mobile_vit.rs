use tch::{Tensor, nn};

use crate::{config::MobileViTConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct MobileViT {
    model: CompactImageTransformer,
}

impl MobileViT {
    pub fn new(vs: &nn::Path, config: MobileViTConfig) -> Self {
        Self {
            model: CompactImageTransformer::new(&(vs / "model"), config),
        }
    }
}

impl nn::ModuleT for MobileViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.model.forward_t(xs, train)
    }
}
