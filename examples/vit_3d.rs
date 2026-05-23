use tch::{Device, Kind, Tensor, nn, nn::ModuleT};
use vit_tch::{ImageSize, ViT3D, ViT3DConfig};

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViT3D::new(
        &vs.root(),
        ViT3DConfig {
            image_size: ImageSize::square(128),
            image_patch_size: ImageSize::square(16),
            frames: 16,
            frame_patch_size: 2,
            num_classes: 1000,
            dim: 1024,
            depth: 6,
            heads: 8,
            mlp_dim: 2048,
            ..Default::default()
        },
    );
    let video = Tensor::randn([4, 3, 16, 128, 128], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&video, false);

    println!("logits shape: {:?}", logits.size());
}
