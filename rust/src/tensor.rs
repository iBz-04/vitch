use tch::Tensor;

pub fn assert_image_patchable(
    image_height: i64,
    image_width: i64,
    patch_height: i64,
    patch_width: i64,
) {
    assert!(
        image_height % patch_height == 0 && image_width % patch_width == 0,
        "image dimensions must be divisible by patch size"
    );
}

pub fn num_patches(
    image_height: i64,
    image_width: i64,
    patch_height: i64,
    patch_width: i64,
) -> i64 {
    assert_image_patchable(image_height, image_width, patch_height, patch_width);
    (image_height / patch_height) * (image_width / patch_width)
}

pub fn patch_dim(channels: i64, patch_height: i64, patch_width: i64) -> i64 {
    channels * patch_height * patch_width
}

pub fn patchify_2d(xs: &Tensor, patch_height: i64, patch_width: i64) -> Tensor {
    let size = xs.size();
    assert_eq!(
        size.len(),
        4,
        "expected image tensor with shape [batch, channels, height, width]"
    );
    let batch = size[0];
    let channels = size[1];
    let height = size[2];
    let width = size[3];
    assert_image_patchable(height, width, patch_height, patch_width);
    let grid_h = height / patch_height;
    let grid_w = width / patch_width;

    xs.view([batch, channels, grid_h, patch_height, grid_w, patch_width])
        .permute([0, 2, 4, 3, 5, 1])
        .contiguous()
        .view([
            batch,
            grid_h * grid_w,
            patch_height * patch_width * channels,
        ])
}

pub fn split_heads(xs: &Tensor, heads: i64) -> Tensor {
    let size = xs.size();
    assert_eq!(
        size.len(),
        3,
        "expected tensor with shape [batch, tokens, channels]"
    );
    let batch = size[0];
    let tokens = size[1];
    let dim = size[2];
    assert_eq!(dim % heads, 0, "channels must be divisible by heads");
    let dim_head = dim / heads;

    xs.view([batch, tokens, heads, dim_head])
        .permute([0, 2, 1, 3])
}

pub fn merge_heads(xs: &Tensor) -> Tensor {
    let size = xs.size();
    assert_eq!(
        size.len(),
        4,
        "expected tensor with shape [batch, heads, tokens, dim_head]"
    );
    let batch = size[0];
    let heads = size[1];
    let tokens = size[2];
    let dim_head = size[3];

    xs.permute([0, 2, 1, 3])
        .contiguous()
        .view([batch, tokens, heads * dim_head])
}

pub fn repeat_token(xs: &Tensor, batch: i64) -> Tensor {
    let size = xs.size();
    match size.as_slice() {
        [tokens, dim] => xs
            .unsqueeze(0)
            .repeat([batch, 1, 1])
            .view([batch, *tokens, *dim]),
        [1, tokens, dim] => xs.repeat([batch, 1, 1]).view([batch, *tokens, *dim]),
        _ => panic!("expected token tensor with shape [tokens, dim] or [1, tokens, dim]"),
    }
}
