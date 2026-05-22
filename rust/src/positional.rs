use tch::{Device, Kind, Tensor};

pub fn posemb_sincos_1d(tokens: i64, dim: i64, temperature: f64, device: Device) -> Tensor {
    assert_eq!(
        dim % 2,
        0,
        "feature dimension must be multiple of 2 for sincos embedding"
    );

    let omega = Tensor::arange(dim / 2, (Kind::Float, device)) / ((dim / 2 - 1) as f64);
    let omega = (omega * temperature.ln()).neg().exp();
    let n = Tensor::arange(tokens, (Kind::Float, device)).unsqueeze(1) * omega.unsqueeze(0);

    Tensor::cat(&[n.sin(), n.cos()], 1)
}

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

pub fn posemb_sincos_3d(
    frames: i64,
    height: i64,
    width: i64,
    dim: i64,
    temperature: f64,
    device: Device,
) -> Tensor {
    let fourier_dim = dim / 6;
    assert!(
        fourier_dim > 1,
        "feature dimension is too small for 3d sincos embedding"
    );

    let omega = Tensor::arange(fourier_dim, (Kind::Float, device)) / ((fourier_dim - 1) as f64);
    let omega = (omega * temperature.ln()).neg().exp();
    let z = Tensor::arange(frames, (Kind::Float, device))
        .view([frames, 1, 1])
        .repeat([1, height, width])
        .flatten(0, -1)
        .unsqueeze(1);
    let y = Tensor::arange(height, (Kind::Float, device))
        .view([1, height, 1])
        .repeat([frames, 1, width])
        .flatten(0, -1)
        .unsqueeze(1);
    let x = Tensor::arange(width, (Kind::Float, device))
        .view([1, 1, width])
        .repeat([frames, height, 1])
        .flatten(0, -1)
        .unsqueeze(1);
    let x = x * omega.unsqueeze(0);
    let y = y * omega.unsqueeze(0);
    let z = z * omega.unsqueeze(0);
    let pe = Tensor::cat(&[x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()], 1);
    let used_dim = fourier_dim * 6;

    if used_dim == dim {
        pe
    } else {
        pe.constant_pad_nd([0, dim - used_dim])
    }
}
