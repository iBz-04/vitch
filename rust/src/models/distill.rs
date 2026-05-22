use tch::{Kind, Tensor, nn};

use crate::{config::DistillConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct Distill {
    student: CompactImageTransformer,
    distill_head: nn::Linear,
    temperature: f64,
}

impl Distill {
    pub fn new(vs: &nn::Path, config: DistillConfig) -> Self {
        let distill_head = nn::linear(
            vs / "distill_head",
            config.student.num_classes,
            config.distill_dim,
            Default::default(),
        );

        Self {
            student: CompactImageTransformer::new(&(vs / "student"), config.student),
            distill_head,
            temperature: config.temperature,
        }
    }

    pub fn forward_t_with_teacher(&self, xs: &Tensor, teacher: &Tensor, train: bool) -> Tensor {
        let logits = self.forward_t(xs, train).apply(&self.distill_head);
        let student_log_prob = (logits / self.temperature).log_softmax(-1, Kind::Float);
        let teacher_prob = (teacher / self.temperature).softmax(-1, Kind::Float);

        -(teacher_prob * student_log_prob)
            .sum_dim_intlist(&[-1_i64][..], false, Kind::Float)
            .mean(Kind::Float)
    }
}

impl nn::ModuleT for Distill {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.student.forward_t(xs, train)
    }
}
