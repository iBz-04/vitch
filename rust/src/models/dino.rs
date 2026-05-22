use tch::{Kind, Tensor, nn, nn::ModuleT};

use crate::{config::DINOConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct DINO {
    student: CompactImageTransformer,
    projection: nn::Linear,
}

impl DINO {
    pub fn new(vs: &nn::Path, config: DINOConfig) -> Self {
        let projection = nn::linear(
            vs / "projection",
            config.num_classes,
            config.num_classes,
            Default::default(),
        );

        Self {
            student: CompactImageTransformer::new(&(vs / "student"), config),
            projection,
        }
    }

    pub fn forward_t_with_teacher(&self, xs: &Tensor, teacher: &Tensor, train: bool) -> Tensor {
        let student = self.forward_t(xs, train).apply(&self.projection);
        let student_log_prob = student.log_softmax(-1, Kind::Float);
        let teacher_prob = teacher.softmax(-1, Kind::Float);

        -(teacher_prob * student_log_prob)
            .sum_dim_intlist(&[-1_i64][..], false, Kind::Float)
            .mean(Kind::Float)
    }
}

impl nn::ModuleT for DINO {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.student.forward_t(xs, train)
    }
}
