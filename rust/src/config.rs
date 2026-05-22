#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    pub height: i64,
    pub width: i64,
}

impl ImageSize {
    pub fn square(size: i64) -> Self {
        Self {
            height: size,
            width: size,
        }
    }

    pub fn new(height: i64, width: i64) -> Self {
        Self { height, width }
    }
}

impl From<i64> for ImageSize {
    fn from(value: i64) -> Self {
        Self::square(value)
    }
}

impl From<(i64, i64)> for ImageSize {
    fn from(value: (i64, i64)) -> Self {
        Self::new(value.0, value.1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pool {
    Cls,
    Mean,
}

#[derive(Debug, Clone)]
pub struct ViTConfig {
    pub image_size: ImageSize,
    pub patch_size: ImageSize,
    pub num_classes: i64,
    pub dim: i64,
    pub depth: usize,
    pub heads: i64,
    pub mlp_dim: i64,
    pub pool: Pool,
    pub channels: i64,
    pub dim_head: i64,
    pub dropout: f64,
    pub emb_dropout: f64,
}

impl Default for ViTConfig {
    fn default() -> Self {
        Self {
            image_size: ImageSize::square(256),
            patch_size: ImageSize::square(32),
            num_classes: 1000,
            dim: 1024,
            depth: 6,
            heads: 16,
            mlp_dim: 2048,
            pool: Pool::Cls,
            channels: 3,
            dim_head: 64,
            dropout: 0.0,
            emb_dropout: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ViTWithDecorrConfig {
    pub vit: ViTConfig,
    pub decorr_sample_frac: f64,
}

impl Default for ViTWithDecorrConfig {
    fn default() -> Self {
        Self {
            vit: ViTConfig::default(),
            decorr_sample_frac: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ViT1DConfig {
    pub seq_len: i64,
    pub patch_size: i64,
    pub num_classes: i64,
    pub dim: i64,
    pub depth: usize,
    pub heads: i64,
    pub mlp_dim: i64,
    pub channels: i64,
    pub dim_head: i64,
    pub dropout: f64,
    pub emb_dropout: f64,
}

impl Default for ViT1DConfig {
    fn default() -> Self {
        Self {
            seq_len: 256,
            patch_size: 16,
            num_classes: 1000,
            dim: 1024,
            depth: 6,
            heads: 8,
            mlp_dim: 2048,
            channels: 3,
            dim_head: 64,
            dropout: 0.0,
            emb_dropout: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ViT3DConfig {
    pub image_size: ImageSize,
    pub image_patch_size: ImageSize,
    pub frames: i64,
    pub frame_patch_size: i64,
    pub num_classes: i64,
    pub dim: i64,
    pub depth: usize,
    pub heads: i64,
    pub mlp_dim: i64,
    pub pool: Pool,
    pub channels: i64,
    pub dim_head: i64,
    pub dropout: f64,
    pub emb_dropout: f64,
}

impl Default for ViT3DConfig {
    fn default() -> Self {
        Self {
            image_size: ImageSize::square(128),
            image_patch_size: ImageSize::square(16),
            frames: 16,
            frame_patch_size: 2,
            num_classes: 1000,
            dim: 1024,
            depth: 6,
            heads: 8,
            mlp_dim: 2048,
            pool: Pool::Cls,
            channels: 3,
            dim_head: 64,
            dropout: 0.0,
            emb_dropout: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MAEConfig {
    pub encoder: ViTConfig,
    pub decoder_dim: i64,
    pub masking_ratio: f64,
    pub decoder_depth: usize,
    pub decoder_heads: i64,
    pub decoder_dim_head: i64,
}

impl Default for MAEConfig {
    fn default() -> Self {
        Self {
            encoder: ViTConfig::default(),
            decoder_dim: 512,
            masking_ratio: 0.75,
            decoder_depth: 1,
            decoder_heads: 8,
            decoder_dim_head: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimMIMConfig {
    pub encoder: ViTConfig,
    pub masking_ratio: f64,
}

impl Default for SimMIMConfig {
    fn default() -> Self {
        Self {
            encoder: ViTConfig::default(),
            masking_ratio: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CCTConfig {
    pub image_size: ImageSize,
    pub num_classes: i64,
    pub embedding_dim: i64,
    pub num_layers: usize,
    pub num_heads: i64,
    pub mlp_ratio: i64,
    pub channels: i64,
    pub kernel_size: i64,
    pub stride: i64,
    pub padding: i64,
    pub pooling_kernel_size: i64,
    pub pooling_stride: i64,
    pub pooling_padding: i64,
    pub dropout: f64,
    pub seq_pool: bool,
}

impl Default for CCTConfig {
    fn default() -> Self {
        Self {
            image_size: ImageSize::square(224),
            num_classes: 1000,
            embedding_dim: 256,
            num_layers: 7,
            num_heads: 4,
            mlp_ratio: 2,
            channels: 3,
            kernel_size: 7,
            stride: 2,
            padding: 3,
            pooling_kernel_size: 3,
            pooling_stride: 2,
            pooling_padding: 1,
            dropout: 0.0,
            seq_pool: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CvTStageConfig {
    pub embed_dim: i64,
    pub embed_kernel: i64,
    pub embed_stride: i64,
    pub proj_kernel: i64,
    pub kv_proj_stride: i64,
    pub heads: i64,
    pub depth: usize,
    pub mlp_mult: i64,
}

impl CvTStageConfig {
    pub fn new(
        embed_dim: i64,
        embed_kernel: i64,
        embed_stride: i64,
        proj_kernel: i64,
        kv_proj_stride: i64,
        heads: i64,
        depth: usize,
        mlp_mult: i64,
    ) -> Self {
        Self {
            embed_dim,
            embed_kernel,
            embed_stride,
            proj_kernel,
            kv_proj_stride,
            heads,
            depth,
            mlp_mult,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CvTConfig {
    pub num_classes: i64,
    pub channels: i64,
    pub stages: [CvTStageConfig; 3],
    pub dim_head: i64,
    pub dropout: f64,
}

impl Default for CvTConfig {
    fn default() -> Self {
        Self {
            num_classes: 1000,
            channels: 3,
            stages: [
                CvTStageConfig::new(64, 7, 4, 3, 2, 1, 1, 4),
                CvTStageConfig::new(192, 3, 2, 3, 2, 3, 2, 4),
                CvTStageConfig::new(384, 3, 2, 3, 2, 6, 10, 4),
            ],
            dim_head: 64,
            dropout: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PiTConfig {
    pub image_size: i64,
    pub patch_size: i64,
    pub num_classes: i64,
    pub dim: i64,
    pub depths: Vec<usize>,
    pub heads: Vec<i64>,
    pub mlp_dim: i64,
    pub channels: i64,
    pub dim_head: i64,
    pub dropout: f64,
    pub emb_dropout: f64,
}

impl Default for PiTConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            num_classes: 1000,
            dim: 256,
            depths: vec![2, 2, 2],
            heads: vec![4, 8, 16],
            mlp_dim: 512,
            channels: 3,
            dim_head: 64,
            dropout: 0.0,
            emb_dropout: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LeViTConfig {
    pub image_size: i64,
    pub num_classes: i64,
    pub dims: Vec<i64>,
    pub depths: Vec<usize>,
    pub heads: Vec<i64>,
    pub mlp_mult: i64,
    pub stages: usize,
    pub channels: i64,
    pub dim_key: i64,
    pub dim_value: i64,
    pub dropout: f64,
    pub num_distill_classes: Option<i64>,
}

impl Default for LeViTConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            num_classes: 1000,
            dims: vec![256, 384, 512],
            depths: vec![4, 4, 4],
            heads: vec![4, 6, 8],
            mlp_mult: 2,
            stages: 3,
            channels: 3,
            dim_key: 32,
            dim_value: 64,
            dropout: 0.0,
            num_distill_classes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompactVisionConfig {
    pub image_size: ImageSize,
    pub patch_size: ImageSize,
    pub num_classes: i64,
    pub dim: i64,
    pub depth: usize,
    pub heads: i64,
    pub mlp_dim: i64,
    pub channels: i64,
    pub dim_head: i64,
    pub dropout: f64,
    pub emb_dropout: f64,
    pub pool: Pool,
}

impl Default for CompactVisionConfig {
    fn default() -> Self {
        Self {
            image_size: ImageSize::square(224),
            patch_size: ImageSize::square(16),
            num_classes: 1000,
            dim: 256,
            depth: 4,
            heads: 4,
            mlp_dim: 512,
            channels: 3,
            dim_head: 64,
            dropout: 0.0,
            emb_dropout: 0.0,
            pool: Pool::Mean,
        }
    }
}

macro_rules! compact_config_alias {
    ($name:ident) => {
        pub type $name = CompactVisionConfig;
    };
}

compact_config_alias!(CaiTConfig);
compact_config_alias!(XCiTConfig);
compact_config_alias!(NesTConfig);
compact_config_alias!(SepViTConfig);
compact_config_alias!(CrossFormerConfig);
compact_config_alias!(TwinsSVTConfig);
compact_config_alias!(RegionViTConfig);
compact_config_alias!(NaViTConfig);
compact_config_alias!(NaViTNestedTensorConfig);
compact_config_alias!(ATSConfig);
compact_config_alias!(MPPConfig);
compact_config_alias!(MP3Config);
compact_config_alias!(DINOConfig);

#[derive(Debug, Clone)]
pub struct MobileViTConfig {
    pub image_size: ImageSize,
    pub num_classes: i64,
    pub dims: [i64; 3],
    pub channels: [i64; 11],
    pub expansion: i64,
    pub kernel_size: i64,
    pub patch_size: ImageSize,
    pub depths: [usize; 3],
    pub dropout: f64,
}

impl Default for MobileViTConfig {
    fn default() -> Self {
        Self {
            image_size: ImageSize::square(256),
            num_classes: 1000,
            dims: [144, 192, 240],
            channels: [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
            expansion: 4,
            kernel_size: 3,
            patch_size: ImageSize::square(2),
            depths: [2, 4, 3],
            dropout: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaxViTConfig {
    pub num_classes: i64,
    pub dim: i64,
    pub depth: Vec<usize>,
    pub dim_head: i64,
    pub dim_conv_stem: Option<i64>,
    pub window_size: i64,
    pub mbconv_expansion_rate: i64,
    pub mbconv_shrinkage_rate: i64,
    pub dropout: f64,
    pub channels: i64,
}

impl Default for MaxViTConfig {
    fn default() -> Self {
        Self {
            num_classes: 1000,
            dim: 64,
            depth: vec![2, 2, 2],
            dim_head: 32,
            dim_conv_stem: None,
            window_size: 7,
            mbconv_expansion_rate: 4,
            mbconv_shrinkage_rate: 4,
            dropout: 0.1,
            channels: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DistillConfig {
    pub student: CompactVisionConfig,
    pub teacher_dim: i64,
    pub distill_dim: i64,
    pub temperature: f64,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            student: CompactVisionConfig::default(),
            teacher_dim: 1000,
            distill_dim: 1000,
            temperature: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ViViTConfig {
    pub image_size: ImageSize,
    pub image_patch_size: ImageSize,
    pub frames: i64,
    pub frame_patch_size: i64,
    pub num_classes: i64,
    pub dim: i64,
    pub depth: usize,
    pub heads: i64,
    pub mlp_dim: i64,
    pub channels: i64,
    pub dim_head: i64,
    pub dropout: f64,
    pub emb_dropout: f64,
}

impl Default for ViViTConfig {
    fn default() -> Self {
        Self {
            image_size: ImageSize::square(128),
            image_patch_size: ImageSize::square(16),
            frames: 8,
            frame_patch_size: 2,
            num_classes: 1000,
            dim: 256,
            depth: 4,
            heads: 4,
            mlp_dim: 512,
            channels: 3,
            dim_head: 64,
            dropout: 0.0,
            emb_dropout: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AcceptVideoWrapperConfig {
    pub image_model: CompactVisionConfig,
    pub frames: i64,
}

impl Default for AcceptVideoWrapperConfig {
    fn default() -> Self {
        Self {
            image_model: CompactVisionConfig::default(),
            frames: 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VATConfig {
    pub image_size: ImageSize,
    pub patch_size: ImageSize,
    pub frames: i64,
    pub action_dim: i64,
    pub num_classes: i64,
    pub dim: i64,
    pub depth: usize,
    pub heads: i64,
    pub mlp_dim: i64,
    pub channels: i64,
    pub dim_head: i64,
    pub dropout: f64,
}

impl Default for VATConfig {
    fn default() -> Self {
        Self {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(16),
            frames: 4,
            action_dim: 8,
            num_classes: 1000,
            dim: 256,
            depth: 4,
            heads: 4,
            mlp_dim: 512,
            channels: 3,
            dim_head: 64,
            dropout: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VAATConfig {
    pub vat: VATConfig,
    pub audio_bins: i64,
    pub audio_frames: i64,
}

impl Default for VAATConfig {
    fn default() -> Self {
        Self {
            vat: VATConfig::default(),
            audio_bins: 32,
            audio_frames: 32,
        }
    }
}
