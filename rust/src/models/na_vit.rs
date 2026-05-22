use tch::{Kind, Tensor, nn};

use crate::{config::NaViTConfig, models::compact::MaskedImageTransformer};

#[derive(Debug)]
pub struct NaViT {
    model: MaskedImageTransformer,
}

impl NaViT {
    pub fn new(vs: &nn::Path, config: NaViTConfig) -> Self {
        Self {
            model: MaskedImageTransformer::new(&(vs / "model"), config),
        }
    }

    pub fn forward_t_with_mask(&self, xs: &Tensor, mask: &Tensor, train: bool) -> Tensor {
        self.model.forward_t_with_mask(xs, Some(mask), train)
    }

    pub fn forward_variable_t(&self, images: &[Tensor], train: bool) -> Tensor {
        assert!(!images.is_empty(), "at least one image is required");
        let stacked = Tensor::stack(images, 0);
        let batch = stacked.size()[0];
        let tokens = self.model.forward_t_with_mask(&stacked, None, train);
        let mask = Tensor::ones([batch, tokens.size()[1]], (Kind::Bool, stacked.device()));

        self.model.forward_t_with_mask(&stacked, Some(&mask), train)
    }
}

impl nn::ModuleT for NaViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.model.forward_t_with_mask(xs, None, train)
    }
}
