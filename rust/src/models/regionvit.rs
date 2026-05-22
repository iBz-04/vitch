use tch::{Tensor, nn};

use crate::{config::RegionViTConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct RegionViT {
    model: CompactImageTransformer,
}

impl RegionViT {
    pub fn new(vs: &nn::Path, config: RegionViTConfig) -> Self {
        Self {
            model: CompactImageTransformer::new(&(vs / "model"), config),
        }
    }
}

impl nn::ModuleT for RegionViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.model.forward_t(xs, train)
    }
}
