use tch::{Device, Kind, Tensor, nn, nn::ModuleT};
use vit_tch::{PiT, PiTConfig};

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = PiT::new(
        &vs.root(),
        PiTConfig {
            image_size: 64,
            patch_size: 8,
            num_classes: 10,
            dim: 64,
            depths: vec![1, 1],
            heads: vec![2, 4],
            mlp_dim: 128,
            dim_head: 32,
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    println!("{:?}", logits.size());
}
