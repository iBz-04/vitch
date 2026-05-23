use tch::{Device, Kind, Tensor, nn, nn::ModuleT};
use vit_tch::{ImageSize, ViViT, ViViTConfig};

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViViT::new(
        &vs.root(),
        ViViTConfig {
            image_size: ImageSize::square(64),
            image_patch_size: ImageSize::square(8),
            frames: 4,
            frame_patch_size: 2,
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            mlp_dim: 256,
            dim_head: 32,
            ..Default::default()
        },
    );
    let video = Tensor::randn([1, 3, 4, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&video, false);

    println!("logits shape: {:?}", logits.size());
}
