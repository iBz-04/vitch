use tch::{Tensor, nn};

use crate::{config::NaViTNestedTensorConfig, models::na_vit::NaViT};

#[derive(Debug)]
pub struct NaViTNestedTensor {
    model: NaViT,
}

impl NaViTNestedTensor {
    pub fn new(vs: &nn::Path, config: NaViTNestedTensorConfig) -> Self {
        Self {
            model: NaViT::new(&(vs / "model"), config),
        }
    }

    pub fn forward_nested_t(&self, images: &[Tensor], train: bool) -> Tensor {
        self.model.forward_variable_t(images, train)
    }
}

impl nn::ModuleT for NaViTNestedTensor {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.model.forward_t(xs, train)
    }
}
