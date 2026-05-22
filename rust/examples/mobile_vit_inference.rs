use tch::{Device, Kind, Tensor, nn, nn::ModuleT};
use vit_tch::{CompactVisionConfig, ImageSize, MobileViT, MobileViTConfig};

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = MobileViT::new(
        &vs.root(),
        MobileViTConfig {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(8),
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            mlp_dim: 256,
            dim_head: 32,
            ..CompactVisionConfig::default()
        },
    );
    let img = Tensor::randn([1, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    println!("{:?}", logits.size());
}
