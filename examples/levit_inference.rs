use tch::{Device, Kind, Tensor, nn, nn::ModuleT};
use vit_tch::{LeViT, LeViTConfig};

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = LeViT::new(
        &vs.root(),
        LeViTConfig {
            image_size: 64,
            num_classes: 10,
            dims: vec![64, 96],
            depths: vec![1, 1],
            heads: vec![2, 3],
            mlp_mult: 2,
            stages: 2,
            dim_key: 16,
            dim_value: 32,
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    println!("logits shape: {:?}", logits.size());
}
