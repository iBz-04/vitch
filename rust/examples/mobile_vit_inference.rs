use tch::{Device, Kind, Tensor, nn, nn::ModuleT};
use vit_tch::{ImageSize, MobileViT, MobileViTConfig};

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = MobileViT::new(
        &vs.root(),
        MobileViTConfig {
            image_size: ImageSize::square(64),
            num_classes: 10,
            dims: [32, 48, 64],
            channels: [8, 8, 16, 16, 24, 24, 32, 32, 48, 48, 96],
            depths: [1, 1, 1],
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    println!("{:?}", logits.size());
}
