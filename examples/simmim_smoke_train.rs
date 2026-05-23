use tch::{Device, Kind, Tensor, nn, nn::ModuleT, nn::OptimizerConfig};
use vit_tch::{ImageSize, SimMIM, SimMIMConfig, ViTConfig};

fn main() -> tch::Result<()> {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = SimMIM::new(
        &vs.root(),
        SimMIMConfig {
            encoder: ViTConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(8),
                num_classes: 0,
                dim: 128,
                depth: 2,
                heads: 4,
                dim_head: 32,
                mlp_dim: 256,
                ..Default::default()
            },
            masking_ratio: 0.5,
        },
    );
    let mut opt = nn::Adam::default().build(&vs, 3e-4)?;
    let images = Tensor::randn([8, 3, 32, 32], (Kind::Float, Device::Cpu));
    let loss = model.forward_t(&images, true);

    opt.backward_step(&loss);
    println!("loss: {:.6}", loss.double_value(&[]));

    Ok(())
}
