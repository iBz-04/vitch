use tch::{Kind, Tensor, nn, nn::ModuleT};

use crate::{config::DINOConfig, models::compact::CompactImageTransformer};

#[derive(Debug)]
pub struct DINO {
    student: CompactImageTransformer,
    projection: nn::Linear,
    student_temp: f64,
    teacher_temp: f64,
}

impl DINO {
    pub fn new(vs: &nn::Path, config: DINOConfig) -> Self {
        assert!(config.student_temp > 0.0, "student temp must be positive");
        assert!(config.teacher_temp > 0.0, "teacher temp must be positive");
        let student_dim = config.student.num_classes;
        let student = CompactImageTransformer::new(&(vs / "student"), config.student);
        let projection = nn::linear(
            vs / "projection",
            student_dim,
            config.projection_dim,
            Default::default(),
        );

        Self {
            student,
            projection,
            student_temp: config.student_temp,
            teacher_temp: config.teacher_temp,
        }
    }

    pub fn forward_t_with_teacher(&self, xs: &Tensor, teacher: &Tensor, train: bool) -> Tensor {
        let student = self.forward_t(xs, train).apply(&self.projection) / self.student_temp;
        let student_log_prob = student.log_softmax(-1, Kind::Float);
        let teacher_prob = (teacher / self.teacher_temp).softmax(-1, Kind::Float);

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
