use tch::{Device, Kind, Tensor, nn, nn::ModuleT};
use vit_tch::{ImageSize, ViT, ViTConfig};

fn main() {
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
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 256, 256], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    println!("logits shape: {:?}", logits.size());
}
