use tch::{Tensor, nn};

use crate::{config::SepViTConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct SepViT {
    model: CompactImageTransformer,
}

impl SepViT {
    pub fn new(vs: &nn::Path, config: SepViTConfig) -> Self {
        Self {
            model: CompactImageTransformer::new(&(vs / "model"), config),
        }
    }
}

impl nn::ModuleT for SepViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.model.forward_t(xs, train)
    }
}
