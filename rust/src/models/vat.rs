use tch::{Kind, Tensor, nn, nn::ModuleT};

use crate::{
    config::{CompactVisionConfig, VATConfig},
    models::compact::CompactImageTransformer,
};

#[derive(Debug)]
pub struct VAT {
    visual: CompactImageTransformer,
    action_proj: nn::Linear,
    head: nn::Linear,
}

impl VAT {
    pub fn new(vs: &nn::Path, config: VATConfig) -> Self {
        let visual_config = CompactVisionConfig {
            image_size: config.image_size,
            patch_size: config.patch_size,
            num_classes: config.dim,
            dim: config.dim,
            depth: config.depth,
            heads: config.heads,
            mlp_dim: config.mlp_dim,
            channels: config.channels,
            dim_head: config.dim_head,
            dropout: config.dropout,
            ..Default::default()
        };
        let action_proj = nn::linear(
            vs / "action_proj",
            config.action_dim,
            config.dim,
            Default::default(),
        );
        let head = nn::linear(
            vs / "head",
            config.dim,
            config.num_classes,
            Default::default(),
        );

        Self {
            visual: CompactImageTransformer::new(&(vs / "visual"), visual_config),
            action_proj,
            head,
        }
    }

    pub fn forward_t_with_actions(&self, video: &Tensor, actions: &Tensor, train: bool) -> Tensor {
        let size = video.size();
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
        let frames_as_batch = video
            .permute([0, 2, 1, 3, 4])
            .contiguous()
            .view([batch * frames, channels, height, width]);
        let visual = self
            .visual
            .forward_t(&frames_as_batch, train)
            .view([batch, frames, -1]);
        let action = actions.apply(&self.action_proj);
        let fused = (visual + action).mean_dim(1, false, Kind::Float);

        fused.apply(&self.head)
    }
}

impl nn::ModuleT for VAT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let batch = xs.size()[0];
        let frames = xs.size()[2];
        let actions = Tensor::zeros([batch, frames, self.action_proj.ws.size()[1]], (xs.kind(), xs.device()));

        self.forward_t_with_actions(xs, &actions, train)
    }
}
