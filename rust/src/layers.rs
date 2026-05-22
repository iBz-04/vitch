use tch::{nn, Tensor};

use crate::tensor::{merge_heads, split_heads};

#[derive(Debug)]
pub struct FeedForward {
    norm: nn::LayerNorm,
    linear1: nn::Linear,
    linear2: nn::Linear,
    dropout: f64,
}

impl FeedForward {
    pub fn new(vs: &nn::Path, dim: i64, hidden_dim: i64, dropout: f64) -> Self {
        let norm = nn::layer_norm(vs / "norm", vec![dim], Default::default());
        let linear1 = nn::linear(vs / "linear1", dim, hidden_dim, Default::default());
        let linear2 = nn::linear(vs / "linear2", hidden_dim, dim, Default::default());

        Self {
            norm,
            linear1,
            linear2,
            dropout,
        }
    }

    pub fn forward_t_with_normed(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        let normed = xs.apply(&self.norm);
        let out = normed
            .apply(&self.linear1)
            .gelu("none")
            .dropout(self.dropout, train)
            .apply(&self.linear2)
            .dropout(self.dropout, train);

        (out, normed)
    }
}

impl nn::ModuleT for FeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward_t_with_normed(xs, train).0
    }
}

#[derive(Debug)]
pub struct Attention {
    norm: nn::LayerNorm,
    to_qkv: nn::Linear,
    to_out: Option<nn::Linear>,
    heads: i64,
    dim_head: i64,
    scale: f64,
    dropout: f64,
}

impl Attention {
    pub fn new(vs: &nn::Path, dim: i64, heads: i64, dim_head: i64, dropout: f64) -> Self {
        let inner_dim = dim_head * heads;
        let norm = nn::layer_norm(vs / "norm", vec![dim], Default::default());
        let to_qkv_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        let to_qkv = nn::linear(vs / "to_qkv", dim, inner_dim * 3, to_qkv_config);
        let to_out = if heads == 1 && dim_head == dim {
            None
        } else {
            Some(nn::linear(vs / "to_out", inner_dim, dim, Default::default()))
        };

        Self {
            norm,
            to_qkv,
            to_out,
            heads,
            dim_head,
            scale: (dim_head as f64).powf(-0.5),
            dropout,
        }
    }

    pub fn forward_t_with_normed(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        let normed = xs.apply(&self.norm);
        let qkv = normed.apply(&self.to_qkv).chunk(3, -1);
        let q = split_heads(&qkv[0], self.heads);
        let k = split_heads(&qkv[1], self.heads);
        let v = split_heads(&qkv[2], self.heads);
        let dots = q.matmul(&k.transpose(-1, -2)) * self.scale;
        let attn = dots.softmax(-1, dots.kind()).dropout(self.dropout, train);
        let out = merge_heads(&attn.matmul(&v));
        let out = match &self.to_out {
            Some(to_out) => out.apply(to_out).dropout(self.dropout, train),
            None => out,
        };

        (out, normed)
    }

    pub fn dim_head(&self) -> i64 {
        self.dim_head
    }
}

impl nn::ModuleT for Attention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward_t_with_normed(xs, train).0
    }
}

#[derive(Debug)]
pub struct TransformerLayer {
    attention: Attention,
    feed_forward: FeedForward,
}

impl TransformerLayer {
    pub fn new(
        vs: &nn::Path,
        dim: i64,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        dropout: f64,
    ) -> Self {
        let attention = Attention::new(&(vs / "attention"), dim, heads, dim_head, dropout);
        let feed_forward = FeedForward::new(&(vs / "feed_forward"), dim, mlp_dim, dropout);

        Self {
            attention,
            feed_forward,
        }
    }

    pub fn forward_t_with_normed(&self, xs: &Tensor, train: bool) -> (Tensor, Vec<Tensor>) {
        let (attn_out, attn_normed) = self.attention.forward_t_with_normed(xs, train);
        let xs = xs + attn_out;
        let (ff_out, ff_normed) = self.feed_forward.forward_t_with_normed(&xs, train);
        let xs = xs + ff_out;

        (xs, vec![attn_normed, ff_normed])
    }
}

impl nn::ModuleT for TransformerLayer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward_t_with_normed(xs, train).0
    }
}

#[derive(Debug)]
pub struct Transformer {
    layers: Vec<TransformerLayer>,
    norm: nn::LayerNorm,
}

impl Transformer {
    pub fn new(
        vs: &nn::Path,
        dim: i64,
        depth: usize,
        heads: i64,
        dim_head: i64,
        mlp_dim: i64,
        dropout: f64,
    ) -> Self {
        let layers = (0..depth)
            .map(|index| {
                let name = format!("layer_{index}");
                TransformerLayer::new(&(vs / name.as_str()), dim, heads, dim_head, mlp_dim, dropout)
            })
            .collect();
        let norm = nn::layer_norm(vs / "norm", vec![dim], Default::default());

        Self { layers, norm }
    }

    pub fn forward_t_with_normed(&self, xs: &Tensor, train: bool) -> (Tensor, Tensor) {
        let mut xs = xs.shallow_clone();
        let mut normed_inputs = Vec::with_capacity(self.layers.len() * 2);

        for layer in &self.layers {
            let (next, mut normed) = layer.forward_t_with_normed(&xs, train);
            xs = next;
            normed_inputs.append(&mut normed);
        }

        (xs.apply(&self.norm), Tensor::stack(&normed_inputs, 0))
    }
}

impl nn::ModuleT for Transformer {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.forward_t_with_normed(xs, train).0
    }
}
