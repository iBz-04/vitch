use tch::{Device, Kind, Tensor, nn, nn::ModuleT};
use vit_tch::{CCT, CCTConfig, ImageSize};

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = CCT::new(
        &vs.root(),
        CCTConfig {
            image_size: ImageSize::square(64),
            num_classes: 10,
            embedding_dim: 128,
            num_layers: 2,
            num_heads: 4,
            mlp_ratio: 2,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            pooling_kernel_size: 3,
            pooling_stride: 2,
            pooling_padding: 1,
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    println!("logits shape: {:?}", logits.size());
}
