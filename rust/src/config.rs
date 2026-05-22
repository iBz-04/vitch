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
