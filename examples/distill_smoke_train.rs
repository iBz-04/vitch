use tch::{Device, Kind, Tensor, nn, nn::OptimizerConfig};
use vit_tch::{CompactVisionConfig, Distill, DistillConfig, ImageSize};

fn main() -> tch::Result<()> {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = Distill::new(
        &vs.root(),
        DistillConfig {
            student: CompactVisionConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(8),
                num_classes: 10,
                dim: 64,
                depth: 1,
                heads: 2,
                mlp_dim: 128,
                dim_head: 32,
                ..Default::default()
            },
            teacher_dim: 10,
            distill_dim: 10,
            temperature: 1.0,
        },
    );
    let mut opt = nn::Adam::default().build(&vs, 3e-4)?;
    let img = Tensor::randn([2, 3, 32, 32], (Kind::Float, Device::Cpu));
    let teacher = Tensor::randn([2, 10], (Kind::Float, Device::Cpu));
    let loss = model.forward_t_with_teacher(&img, &teacher, true);

    opt.backward_step(&loss);
    println!("loss: {:.6}", loss.double_value(&[]));

    Ok(())
}
