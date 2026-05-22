use tch::{Kind, Tensor, nn};

use crate::{config::AcceptVideoWrapperConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct AcceptVideoWrapper {
    image_model: CompactImageTransformer,
    time_pos_emb: Option<Tensor>,
    embed_proj: Option<nn::Linear>,
    frames: i64,
}

impl AcceptVideoWrapper {
    pub fn new(vs: &nn::Path, config: AcceptVideoWrapperConfig) -> Self {
        assert!(config.frames > 0, "frames must be positive");
        let image_dim = config.image_model.num_classes;
        let time_pos_emb = config.add_time_pos_emb.then(|| {
            vs.var(
                "time_pos_emb",
                &[config.frames, config.proj_embed_to_dim.unwrap_or(image_dim)],
                nn::Init::Randn {
                    mean: 0.0,
                    stdev: 1e-2,
                },
            )
        });
        let embed_proj = config
            .proj_embed_to_dim
            .map(|dim| nn::linear(vs / "embed_proj", image_dim, dim, Default::default()));

        Self {
            image_model: CompactImageTransformer::new(&(vs / "image_model"), config.image_model),
            time_pos_emb,
            embed_proj,
            frames: config.frames,
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
        assert!(
            frames <= self.frames,
            "received more frames than configured time sequence length"
        );
        let frames_as_batch = xs.permute([0, 2, 1, 3, 4]).contiguous().view([
            batch * frames,
            channels,
            height,
            width,
        ]);
        let mut logits = self.image_model.forward_t(&frames_as_batch, train);
        if let Some(embed_proj) = &self.embed_proj {
            logits = logits.apply(embed_proj);
        }
        let dim = logits.size()[1];
        let mut logits = logits.view([batch, frames, dim]);
        if let Some(time_pos_emb) = &self.time_pos_emb {
            logits += time_pos_emb.narrow(0, 0, frames).unsqueeze(0);
        }

        logits.mean_dim(1, false, Kind::Float)
    }
}
