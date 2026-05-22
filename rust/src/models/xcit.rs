use tch::{Tensor, nn};

use crate::{config::XCiTConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct XCiT {
    model: CompactImageTransformer,
}

impl XCiT {
    pub fn new(vs: &nn::Path, config: XCiTConfig) -> Self {
        Self {
            model: CompactImageTransformer::new(&(vs / "model"), config),
        }
    }
}

impl nn::ModuleT for XCiT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.model.forward_t(xs, train)
    }
}
