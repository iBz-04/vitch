use tch::{Kind, Tensor, nn};

use crate::{config::VAATConfig, models::vat::VAT};

#[derive(Debug)]
pub struct VAAT {
    vat: VAT,
    audio_proj: nn::Linear,
}

impl VAAT {
    pub fn new(vs: &nn::Path, config: VAATConfig) -> Self {
        let audio_proj = nn::linear(
            vs / "audio_proj",
            config.audio_bins * config.audio_frames,
            config.vat.num_classes,
            Default::default(),
        );

        Self {
            vat: VAT::new(&(vs / "vat"), config.vat),
            audio_proj,
        }
    }

    pub fn forward_t_with_audio_actions(
        &self,
        video: &Tensor,
        audio: &Tensor,
        actions: &Tensor,
        train: bool,
    ) -> Tensor {
        let visual = self.vat.forward_t_with_actions(video, actions, train);
        let audio = audio.flatten(1, -1).apply(&self.audio_proj);

        (visual + audio).softmax(-1, Kind::Float)
    }
}

impl nn::ModuleT for VAAT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let batch = xs.size()[0];
        let frames = xs.size()[2];
        let actions = Tensor::zeros([batch, frames, 8], (xs.kind(), xs.device()));
        let audio = Tensor::zeros([batch, 32, 32], (xs.kind(), xs.device()));

        self.forward_t_with_audio_actions(xs, &audio, &actions, train)
    }
}
