use tch::{Tensor, nn};

use crate::{
    config::MobileViTConfig, conv_layers::ConvBnSiLU, layers::Transformer,
    tensor::assert_image_patchable,
};

#[derive(Debug)]
struct MV2Block {
    expand: Option<ConvBnSiLU>,
    depthwise: ConvBnSiLU,
    project_conv: nn::Conv2D,
    project_bn: nn::BatchNorm,
    residual: bool,
}

impl MV2Block {
    fn new(vs: &nn::Path, dim_in: i64, dim_out: i64, stride: i64, expansion: i64) -> Self {
        assert!(stride == 1 || stride == 2, "stride must be 1 or 2");
        assert!(expansion > 0, "expansion must be positive");
        let hidden_dim = dim_in * expansion;
        let expand = if expansion == 1 {
            None
        } else {
            Some(ConvBnSiLU::new(
                &(vs / "expand"),
                dim_in,
                hidden_dim,
                1,
                1,
                0,
                1,
            ))
        };
        let depthwise = ConvBnSiLU::new(
            &(vs / "depthwise"),
            hidden_dim,
            hidden_dim,
            3,
            stride,
            1,
            hidden_dim,
        );
        let project_conv = nn::conv2d(
            vs / "project_conv",
            hidden_dim,
            dim_out,
            1,
            nn::ConvConfig {
                bias: false,
                ..Default::default()
            },
        );
        let project_bn = nn::batch_norm2d(vs / "project_bn", dim_out, Default::default());
        let residual = stride == 1 && dim_in == dim_out;

        Self {
            expand,
            depthwise,
            project_conv,
            project_bn,
            residual,
        }
    }
}

impl nn::ModuleT for MV2Block {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut out = match &self.expand {
            Some(expand) => expand.forward_t(xs, train),
            None => xs.shallow_clone(),
        };
        out = self.depthwise.forward_t(&out, train);
        out = out
            .apply(&self.project_conv)
            .apply_t(&self.project_bn, train);

        if self.residual { xs + out } else { out }
    }
}

#[derive(Debug)]
struct MobileViTBlock {
    conv1: ConvBnSiLU,
    conv2: ConvBnSiLU,
    transformer: Transformer,
    conv3: ConvBnSiLU,
    conv4: ConvBnSiLU,
    patch_height: i64,
    patch_width: i64,
}

impl MobileViTBlock {
    fn new(
        vs: &nn::Path,
        dim: i64,
        depth: usize,
        channel: i64,
        kernel_size: i64,
        patch_height: i64,
        patch_width: i64,
        mlp_dim: i64,
        dropout: f64,
    ) -> Self {
        let padding = kernel_size / 2;
        let conv1 = ConvBnSiLU::new(
            &(vs / "conv1"),
            channel,
            channel,
            kernel_size,
            1,
            padding,
            1,
        );
        let conv2 = ConvBnSiLU::new(&(vs / "conv2"), channel, dim, 1, 1, 0, 1);
        let transformer =
            Transformer::new(&(vs / "transformer"), dim, depth, 4, 8, mlp_dim, dropout);
        let conv3 = ConvBnSiLU::new(&(vs / "conv3"), dim, channel, 1, 1, 0, 1);
        let conv4 = ConvBnSiLU::new(
            &(vs / "conv4"),
            channel * 2,
            channel,
            kernel_size,
            1,
            padding,
            1,
        );

        Self {
            conv1,
            conv2,
            transformer,
            conv3,
            conv4,
            patch_height,
            patch_width,
        }
    }

    fn fold_patches(&self, xs: &Tensor) -> Tensor {
        let size = xs.size();
        let batch = size[0];
        let dim = size[1];
        let height = size[2];
        let width = size[3];
        assert_image_patchable(height, width, self.patch_height, self.patch_width);
        let grid_h = height / self.patch_height;
        let grid_w = width / self.patch_width;

        xs.view([
            batch,
            dim,
            grid_h,
            self.patch_height,
            grid_w,
            self.patch_width,
        ])
        .permute([0, 3, 5, 2, 4, 1])
        .contiguous()
        .view([
            batch * self.patch_height * self.patch_width,
            grid_h * grid_w,
            dim,
        ])
    }

    fn unfold_patches(&self, xs: &Tensor, batch: i64, height: i64, width: i64) -> Tensor {
        let dim = xs.size()[2];
        let grid_h = height / self.patch_height;
        let grid_w = width / self.patch_width;

        xs.view([
            batch,
            self.patch_height,
            self.patch_width,
            grid_h,
            grid_w,
            dim,
        ])
        .permute([0, 5, 3, 1, 4, 2])
        .contiguous()
        .view([batch, dim, height, width])
    }
}

impl nn::ModuleT for MobileViTBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let residual = xs.shallow_clone();
        let mut out = self.conv1.forward_t(xs, train);
        out = self.conv2.forward_t(&out, train);

        let size = out.size();
        let batch = size[0];
        let height = size[2];
        let width = size[3];
        out = self.fold_patches(&out);
        out = self.transformer.forward_t(&out, train);
        out = self.unfold_patches(&out, batch, height, width);

        out = self.conv3.forward_t(&out, train);
        out = Tensor::cat(&[out, residual], 1);
        self.conv4.forward_t(&out, train)
    }
}

#[derive(Debug)]
pub struct MobileViT {
    conv1: ConvBnSiLU,
    stem: Vec<MV2Block>,
    trunk: Vec<(MV2Block, MobileViTBlock)>,
    to_logits_conv: ConvBnSiLU,
    classifier: nn::Linear,
}

impl MobileViT {
    pub fn new(vs: &nn::Path, config: MobileViTConfig) -> Self {
        assert_image_patchable(
            config.image_size.height,
            config.image_size.width,
            config.patch_size.height,
            config.patch_size.width,
        );
        let channels = config.channels;
        let dims = config.dims;
        let depths = config.depths;
        let conv1 = ConvBnSiLU::new(
            &(vs / "conv1"),
            3,
            channels[0],
            config.kernel_size,
            2,
            config.kernel_size / 2,
            1,
        );
        let stem = vec![
            MV2Block::new(
                &(vs / "stem_0"),
                channels[0],
                channels[1],
                1,
                config.expansion,
            ),
            MV2Block::new(
                &(vs / "stem_1"),
                channels[1],
                channels[2],
                2,
                config.expansion,
            ),
            MV2Block::new(
                &(vs / "stem_2"),
                channels[2],
                channels[3],
                1,
                config.expansion,
            ),
            MV2Block::new(
                &(vs / "stem_3"),
                channels[3],
                channels[3],
                1,
                config.expansion,
            ),
        ];
        let trunk = vec![
            (
                MV2Block::new(
                    &(vs / "trunk_0_conv"),
                    channels[3],
                    channels[4],
                    2,
                    config.expansion,
                ),
                MobileViTBlock::new(
                    &(vs / "trunk_0_attn"),
                    dims[0],
                    depths[0],
                    channels[5],
                    config.kernel_size,
                    config.patch_size.height,
                    config.patch_size.width,
                    dims[0] * 2,
                    config.dropout,
                ),
            ),
            (
                MV2Block::new(
                    &(vs / "trunk_1_conv"),
                    channels[5],
                    channels[6],
                    2,
                    config.expansion,
                ),
                MobileViTBlock::new(
                    &(vs / "trunk_1_attn"),
                    dims[1],
                    depths[1],
                    channels[7],
                    config.kernel_size,
                    config.patch_size.height,
                    config.patch_size.width,
                    dims[1] * 4,
                    config.dropout,
                ),
            ),
            (
                MV2Block::new(
                    &(vs / "trunk_2_conv"),
                    channels[7],
                    channels[8],
                    2,
                    config.expansion,
                ),
                MobileViTBlock::new(
                    &(vs / "trunk_2_attn"),
                    dims[2],
                    depths[2],
                    channels[9],
                    config.kernel_size,
                    config.patch_size.height,
                    config.patch_size.width,
                    dims[2] * 4,
                    config.dropout,
                ),
            ),
        ];
        let to_logits_conv = ConvBnSiLU::new(
            &(vs / "to_logits_conv"),
            channels[9],
            channels[10],
            1,
            1,
            0,
            1,
        );
        let classifier = nn::linear(
            vs / "classifier",
            channels[10],
            config.num_classes,
            nn::LinearConfig {
                bias: false,
                ..Default::default()
            },
        );

        Self {
            conv1,
            stem,
            trunk,
            to_logits_conv,
            classifier,
        }
    }
}

impl nn::ModuleT for MobileViT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut xs = self.conv1.forward_t(xs, train);

        for block in &self.stem {
            xs = block.forward_t(&xs, train);
        }

        for (conv, attn) in &self.trunk {
            xs = conv.forward_t(&xs, train);
            xs = attn.forward_t(&xs, train);
        }

        xs = self.to_logits_conv.forward_t(&xs, train);
        xs.adaptive_avg_pool2d([1, 1])
            .flatten(1, -1)
            .apply(&self.classifier)
    }
}
