use tch::{Device, Kind, Tensor, nn, nn::ModuleT};
use vit_tch::{ViT1D, ViT1DConfig};

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViT1D::new(
        &vs.root(),
        ViT1DConfig {
            seq_len: 256,
            patch_size: 16,
            num_classes: 1000,
            dim: 1024,
            depth: 6,
            heads: 8,
            mlp_dim: 2048,
            ..Default::default()
        },
    );
    let series = Tensor::randn([4, 3, 256], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&series, false);

    println!("logits shape: {:?}", logits.size());
}
