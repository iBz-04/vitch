use tch::{Kind, Tensor, nn};

use crate::{config::AcceptVideoWrapperConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct AcceptVideoWrapper {
    image_model: CompactImageTransformer,
}

impl AcceptVideoWrapper {
    pub fn new(vs: &nn::Path, config: AcceptVideoWrapperConfig) -> Self {
        Self {
            image_model: CompactImageTransformer::new(&(vs / "image_model"), config.image_model),
        }
    }
}

impl nn::ModuleT for AcceptVideoWrapper {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        assert_eq!(
            size.len(),
            5,
            "expected video tensor with shape [batch, channels, frames, height, width]"
        );
        let batch = size[0];
        let channels = size[1];
        let frames = size[2];
        let height = size[3];
        let width = size[4];
        let frames_as_batch = xs.permute([0, 2, 1, 3, 4]).contiguous().view([
            batch * frames,
            channels,
            height,
            width,
        ]);
        let logits = self.image_model.forward_t(&frames_as_batch, train);

        logits
            .view([batch, frames, logits.size()[1]])
            .mean_dim(1, false, Kind::Float)
    }
}
