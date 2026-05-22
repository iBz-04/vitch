use tch::{Kind, Tensor};

#[derive(Debug)]
pub struct RandomMask {
    pub masked_indices: Tensor,
    pub unmasked_indices: Tensor,
}

impl RandomMask {
    pub fn new(batch: i64, tokens: i64, masking_ratio: f64, device: tch::Device) -> Self {
        assert!(
            (0.0..1.0).contains(&masking_ratio),
            "masking ratio must be between 0 and 1"
        );
        let num_masked = 1.max((tokens as f64 * masking_ratio) as i64);
        assert!(num_masked < tokens, "at least one token must remain unmasked");
        let indices = Tensor::rand([batch, tokens], (Kind::Float, device)).argsort(-1, false);
        let masked_indices = indices.narrow(1, 0, num_masked);
        let unmasked_indices = indices.narrow(1, num_masked, tokens - num_masked);

        Self {
            masked_indices,
            unmasked_indices,
        }
    }
}

pub fn gather_tokens(tokens: &Tensor, indices: &Tensor) -> Tensor {
    let dim = tokens.size()[2];
    let indices = indices.unsqueeze(-1).repeat([1, 1, dim]);

    tokens.gather(1, &indices, false)
}

pub fn scatter_tokens(batch: i64, tokens: i64, dim: i64, indices: &Tensor, values: &Tensor) -> Tensor {
    let target = Tensor::zeros([batch, tokens, dim], (values.kind(), values.device()));
    let indices = indices.unsqueeze(-1).repeat([1, 1, dim]);

    target.scatter(1, &indices, values)
}
