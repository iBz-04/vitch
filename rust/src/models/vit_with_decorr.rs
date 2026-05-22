use tch::{IndexOp, Kind, Tensor, nn};

use crate::{
    config::{Pool, ViTWithDecorrConfig},
    layers::Transformer,
    models::vit::PatchEmbedding,
    tensor::{num_patches, patch_dim, repeat_token},
};

#[derive(Debug, Clone)]
pub struct DecorrelationLoss {
    sample_frac: f64,
    soft_validate_num_sampled: bool,
}

impl DecorrelationLoss {
    pub fn new(sample_frac: f64) -> Self {
        Self::with_soft_validation(sample_frac, false)
    }

    pub fn with_soft_validation(sample_frac: f64, soft_validate_num_sampled: bool) -> Self {
        assert!((0.0..=1.0).contains(&sample_frac));
        Self {
            sample_frac,
            soft_validate_num_sampled,
        }
    }

    pub fn forward(&self, tokens: &Tensor) -> Tensor {
        let mut tokens = tokens.shallow_clone();
        let size = tokens.size();
        let seq_len = size[size.len() - 2];
        let dim = size[size.len() - 1];

        if self.sample_frac < 1.0 {
            let num_sampled = (seq_len as f64 * self.sample_frac) as i64;
            assert!(self.soft_validate_num_sampled || num_sampled >= 2);

            if num_sampled <= 1 {
                return Tensor::zeros([], (tokens.kind(), tokens.device()));
            }

            tokens = sample_tokens(&tokens, num_sampled);
        }

        let dist =
            tokens.transpose(-1, -2).matmul(&tokens) / (tokens.size()[tokens.dim() - 2] as f64);
        let eye = Tensor::eye(dim, (tokens.kind(), tokens.device()));
        let loss = dist.pow_tensor_scalar(2.0) * (Tensor::ones_like(&eye) - eye)
            / ((dim - 1) * dim) as f64;
        loss.sum_dim_intlist(&[-1, -2], false, Kind::Float)
            .mean(Kind::Float)
    }
}

fn sample_tokens(tokens: &Tensor, num_sampled: i64) -> Tensor {
    let size = tokens.size();
    let dim = size[size.len() - 1];
    let seq_len = size[size.len() - 2];
    let outer: i64 = size[..size.len() - 2].iter().product();
    let flat = tokens.contiguous().view([outer, seq_len, dim]);
    let indices = Tensor::rand([outer, seq_len], (tokens.kind(), tokens.device()))
        .argsort(-1, false)
        .i((.., 0..num_sampled))
        .unsqueeze(-1)
        .repeat([1, 1, dim]);
    let sampled = flat.gather(1, &indices, false);
    let mut new_size = size[..size.len() - 2].to_vec();
    new_size.push(num_sampled);
    new_size.push(dim);

    sampled.view(new_size.as_slice())
}

#[derive(Debug)]
pub struct ViTWithDecorr {
    patch_embedding: PatchEmbedding,
    cls_token: Tensor,
    pos_embedding: Tensor,
    transformer: Transformer,
    mlp_head: nn::Linear,
    pool: Pool,
    emb_dropout: f64,
    decorr_loss: Option<DecorrelationLoss>,
}

impl ViTWithDecorr {
    pub fn new(vs: &nn::Path, config: ViTWithDecorrConfig) -> Self {
        let vit = config.vit;
        let image_height = vit.image_size.height;
        let image_width = vit.image_size.width;
        let patch_height = vit.patch_size.height;
        let patch_width = vit.patch_size.width;
        let num_patches = num_patches(image_height, image_width, patch_height, patch_width);
        let patch_dim = patch_dim(vit.channels, patch_height, patch_width);
        let patch_embedding = PatchEmbedding::new(
            &(vs / "patch_embedding"),
            patch_dim,
            vit.dim,
            patch_height,
            patch_width,
        );
        let pos_embedding = vs.var(
            "pos_embedding",
            &[1, num_patches + 1, vit.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let cls_token = vs.var(
            "cls_token",
            &[1, 1, vit.dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        );
        let transformer = Transformer::new(
            &(vs / "transformer"),
            vit.dim,
            vit.depth,
            vit.heads,
            vit.dim_head,
            vit.mlp_dim,
            vit.dropout,
        );
        let mlp_head = nn::linear(
            vs / "mlp_head",
            vit.dim,
            vit.num_classes,
            Default::default(),
        );
        let decorr_loss = if config.decorr_sample_frac > 0.0 {
            Some(DecorrelationLoss::new(config.decorr_sample_frac))
        } else {
            None
        };

        Self {
            patch_embedding,
            cls_token,
            pos_embedding,
            transformer,
            mlp_head,
            pool: vit.pool,
            emb_dropout: vit.emb_dropout,
            decorr_loss,
        }
    }

    pub fn forward_t_with_aux(
        &self,
        xs: &Tensor,
        train: bool,
        return_decorr_aux_loss: bool,
    ) -> (Tensor, Tensor) {
        let mut xs = xs.apply(&self.patch_embedding);
        let batch = xs.size()[0];
        let tokens = xs.size()[1];
        let cls_tokens = repeat_token(&self.cls_token, batch);
        xs = Tensor::cat(&[cls_tokens, xs], 1);
        xs = (xs + self.pos_embedding.i((.., 0..tokens + 1, ..))).dropout(self.emb_dropout, train);

        let (xs, normed_layer_inputs) = self.transformer.forward_t_with_normed(&xs, train);
        let aux_loss = if return_decorr_aux_loss {
            match &self.decorr_loss {
                Some(decorr_loss) => decorr_loss.forward(&normed_layer_inputs),
                None => Tensor::zeros([], (xs.kind(), xs.device())),
            }
        } else {
            Tensor::zeros([], (xs.kind(), xs.device()))
        };
        let xs = match self.pool {
            Pool::Mean => xs.mean_dim(1, false, xs.kind()),
            Pool::Cls => xs.i((.., 0)),
        };
        let logits = xs.apply(&self.mlp_head);

        (logits, aux_loss)
    }
}

impl nn::ModuleT for ViTWithDecorr {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward_t_with_aux(xs, train, train).0
    }
}
