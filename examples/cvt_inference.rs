use tch::{Device, Kind, Tensor, nn, nn::ModuleT};
use vit_tch::{CvT, CvTConfig, CvTStageConfig};

fn main() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = CvT::new(
        &vs.root(),
        CvTConfig {
            num_classes: 10,
            stages: [
                CvTStageConfig::new(32, 3, 2, 3, 1, 1, 1, 2),
                CvTStageConfig::new(64, 3, 2, 3, 1, 2, 1, 2),
                CvTStageConfig::new(128, 3, 2, 3, 1, 4, 1, 2),
            ],
            dim_head: 32,
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    println!("{:?}", logits.size());
}
