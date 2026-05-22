use tch::{Tensor, nn};

use crate::{config::CaiTConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct CaiT {
    model: CompactImageTransformer,
}

impl CaiT {
    pub fn new(vs: &nn::Path, config: CaiTConfig) -> Self {
        Self {
            model: CompactImageTransformer::new(&(vs / "model"), config),
        }
    }
}

impl nn::ModuleT for CaiT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.model.forward_t(xs, train)
    }
}
