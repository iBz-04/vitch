use tch::{Device, Kind, Tensor};

pub fn posemb_sincos_2d(
    height: i64,
    width: i64,
    dim: i64,
    temperature: f64,
    device: Device,
) -> Tensor {
    assert_eq!(
        dim % 4,
        0,
        "feature dimension must be multiple of 4 for sincos embedding"
    );

    let omega = Tensor::arange(dim / 4, (Kind::Float, device)) / ((dim / 4 - 1) as f64);
    let omega = (omega * temperature.ln()).neg().exp();
    let y = Tensor::arange(height, (Kind::Float, device))
        .view([height, 1])
        .repeat([1, width])
        .flatten(0, -1)
        .unsqueeze(1);
    let x = Tensor::arange(width, (Kind::Float, device))
        .view([1, width])
        .repeat([height, 1])
        .flatten(0, -1)
        .unsqueeze(1);
    let x = x * omega.unsqueeze(0);
    let y = y * omega.unsqueeze(0);

    Tensor::cat(&[x.sin(), x.cos(), y.sin(), y.cos()], 1)
}
