use tch::{Tensor, nn};

use crate::{config::MaxViTConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct MaxViT {
    model: CompactImageTransformer,
}

impl MaxViT {
    pub fn new(vs: &nn::Path, config: MaxViTConfig) -> Self {
        Self {
            model: CompactImageTransformer::new(&(vs / "model"), config),
        }
    }
}

impl nn::ModuleT for MaxViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.model.forward_t(xs, train)
    }
}
