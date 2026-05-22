use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor, nn};
use vit_tch::{ImageSize, Pool, SimpleViT, ViT, ViTConfig, ViTWithDecorr, ViTWithDecorrConfig};

#[test]
fn vit_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(256),
            patch_size: ImageSize::square(32),
            num_classes: 1000,
            dim: 1024,
            depth: 6,
            heads: 16,
            mlp_dim: 2048,
            dropout: 0.1,
            emb_dropout: 0.1,
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 256, 256], (Kind::Float, Device::Cpu));
    let preds = model.forward_t(&img, false);

    assert_eq!(preds.size(), [1, 1000]);
}

#[test]
fn vit_mean_pool_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(16),
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            mlp_dim: 256,
            pool: Pool::Mean,
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let preds = model.forward_t(&img, false);

    assert_eq!(preds.size(), [2, 10]);
}

#[test]
fn vit_rectangular_inputs_work() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::new(256, 128),
            patch_size: ImageSize::new(32, 16),
            num_classes: 1000,
            dim: 256,
            depth: 2,
            heads: 8,
            mlp_dim: 512,
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 256, 128], (Kind::Float, Device::Cpu));
    let preds = model.forward_t(&img, false);

    assert_eq!(preds.size(), [1, 1000]);
}

#[test]
fn simple_vit_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = SimpleViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(256),
            patch_size: ImageSize::square(32),
            num_classes: 1000,
            dim: 1024,
            depth: 6,
            heads: 16,
            mlp_dim: 2048,
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 256, 256], (Kind::Float, Device::Cpu));
    let preds = model.forward_t(&img, false);

    assert_eq!(preds.size(), [1, 1000]);
}

#[test]
fn vit_with_decorr_outputs_logits_and_aux_loss() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViTWithDecorr::new(
        &vs.root(),
        ViTWithDecorrConfig {
            vit: ViTConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(4),
                num_classes: 100,
                dim: 128,
                depth: 2,
                heads: 8,
                dim_head: 16,
                mlp_dim: 512,
                ..Default::default()
            },
            decorr_sample_frac: 1.0,
        },
    );
    let img = Tensor::randn([2, 3, 32, 32], (Kind::Float, Device::Cpu));
    let (logits, aux_loss) = model.forward_t_with_aux(&img, true, true);

    assert_eq!(logits.size(), [2, 100]);
    assert_eq!(aux_loss.size(), [] as [i64; 0]);
}

#[test]
#[should_panic(expected = "image dimensions must be divisible by patch size")]
fn vit_rejects_invalid_patch_size() {
    let vs = nn::VarStore::new(Device::Cpu);
    let _ = ViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::new(255, 256),
            patch_size: ImageSize::square(32),
            ..Default::default()
        },
    );
}
