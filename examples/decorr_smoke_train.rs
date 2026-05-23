use tch::{Device, Kind, Tensor, nn, nn::OptimizerConfig};
use vit_tch::{ImageSize, ViTConfig, ViTWithDecorr, ViTWithDecorrConfig};

fn main() -> tch::Result<()> {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViTWithDecorr::new(
        &vs.root(),
        ViTWithDecorrConfig {
            vit: ViTConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(4),
                num_classes: 100,
                dim: 128,
                depth: 6,
                heads: 8,
                dim_head: 64,
                mlp_dim: 512,
                ..Default::default()
            },
            decorr_sample_frac: 1.0,
        },
    );
    let mut opt = nn::Adam::default().build(&vs, 3e-4)?;
    let images = Tensor::randn([32, 3, 32, 32], (Kind::Float, Device::Cpu));
    let labels = Tensor::randint(100, [32], (Kind::Int64, Device::Cpu));
    let (logits, decorr_loss) = model.forward_t_with_aux(&images, true, true);
    let loss = logits.cross_entropy_for_logits(&labels);
    let total_loss = loss + decorr_loss * 0.1;

    opt.backward_step(&total_loss);
    println!("loss: {:.6}", total_loss.double_value(&[]));

    Ok(())
}
