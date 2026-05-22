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

pub fn assert_sequence_patchable(seq_len: i64, patch_size: i64) {
    assert!(
        seq_len % patch_size == 0,
        "sequence length must be divisible by patch size"
    );
}

pub fn assert_video_patchable(
    frames: i64,
    frame_patch_size: i64,
    image_height: i64,
    image_width: i64,
    patch_height: i64,
    patch_width: i64,
) {
    assert_image_patchable(image_height, image_width, patch_height, patch_width);
    assert!(
        frames % frame_patch_size == 0,
        "frames must be divisible by frame patch size"
    );
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

pub fn patchify_1d(xs: &Tensor, patch_size: i64) -> Tensor {
    let size = xs.size();
    assert_eq!(
        size.len(),
        3,
        "expected sequence tensor with shape [batch, channels, length]"
    );
    let batch = size[0];
    let channels = size[1];
    let seq_len = size[2];
    assert_sequence_patchable(seq_len, patch_size);
    let tokens = seq_len / patch_size;

    xs.view([batch, channels, tokens, patch_size])
        .permute([0, 2, 3, 1])
        .contiguous()
        .view([batch, tokens, patch_size * channels])
}

pub fn patchify_3d_flat(
    xs: &Tensor,
    frame_patch_size: i64,
    patch_height: i64,
    patch_width: i64,
) -> Tensor {
    let patches = patchify_3d_grid(xs, frame_patch_size, patch_height, patch_width);
    let size = patches.size();
    let batch = size[0];
    let frame_tokens = size[1];
    let height_tokens = size[2];
    let width_tokens = size[3];
    let dim = size[4];

    patches.view([batch, frame_tokens * height_tokens * width_tokens, dim])
}

pub fn patchify_3d_grid(
    xs: &Tensor,
    frame_patch_size: i64,
    patch_height: i64,
    patch_width: i64,
) -> Tensor {
    let size = xs.size();
    assert_eq!(
        size.len(),
        5,
        "expected video tensor with shape [batch, channels, frames, height, width]"
    );
    let batch = size[0];
    let channels = size[1];
    let frames = size[2];
    let height = size[3];
    let width = size[4];
    assert_video_patchable(
        frames,
        frame_patch_size,
        height,
        width,
        patch_height,
        patch_width,
    );
    let frame_tokens = frames / frame_patch_size;
    let height_tokens = height / patch_height;
    let width_tokens = width / patch_width;

    xs.view(
        &[
            batch,
            channels,
            frame_tokens,
            frame_patch_size,
            height_tokens,
            patch_height,
            width_tokens,
            patch_width,
        ][..],
    )
    .permute([0, 2, 4, 6, 3, 5, 7, 1])
    .contiguous()
    .view([
        batch,
        frame_tokens,
        height_tokens,
        width_tokens,
        frame_patch_size * patch_height * patch_width * channels,
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

pub fn image_to_tokens(xs: &Tensor) -> Tensor {
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

    xs.permute([0, 2, 3, 1])
        .contiguous()
        .view([batch, height * width, channels])
}

pub fn tokens_to_image(xs: &Tensor, height: i64, width: i64) -> Tensor {
    let size = xs.size();
    assert_eq!(
        size.len(),
        3,
        "expected token tensor with shape [batch, tokens, channels]"
    );
    let batch = size[0];
    let tokens = size[1];
    let channels = size[2];
    assert_eq!(tokens, height * width, "tokens must match image grid");

    xs.view([batch, height, width, channels])
        .permute([0, 3, 1, 2])
        .contiguous()
}

pub fn window_partition(xs: &Tensor, window_height: i64, window_width: i64) -> Tensor {
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
    assert_image_patchable(height, width, window_height, window_width);
    let grid_h = height / window_height;
    let grid_w = width / window_width;

    xs.view([
        batch,
        channels,
        grid_h,
        window_height,
        grid_w,
        window_width,
    ])
    .permute([0, 2, 4, 1, 3, 5])
    .contiguous()
    .view([
        batch * grid_h * grid_w,
        channels,
        window_height,
        window_width,
    ])
}

pub fn window_unpartition(
    windows: &Tensor,
    batch: i64,
    height: i64,
    width: i64,
    window_height: i64,
    window_width: i64,
) -> Tensor {
    let size = windows.size();
    assert_eq!(
        size.len(),
        4,
        "expected window tensor with shape [windows, channels, height, width]"
    );
    let channels = size[1];
    assert_image_patchable(height, width, window_height, window_width);
    let grid_h = height / window_height;
    let grid_w = width / window_width;
    assert_eq!(
        size[0],
        batch * grid_h * grid_w,
        "window count must match batch and image grid"
    );

    windows
        .view([
            batch,
            grid_h,
            grid_w,
            channels,
            window_height,
            window_width,
        ])
        .permute([0, 3, 1, 4, 2, 5])
        .contiguous()
        .view([batch, channels, height, width])
}
