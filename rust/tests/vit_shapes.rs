use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor, nn};
use vit_tch::{
    ATS, ATSConfig, AcceptVideoWrapper, AcceptVideoWrapperConfig, CCT, CCTConfig, CaiT, CaiTConfig,
    CompactVisionConfig, CrossFormer, CrossFormerConfig, CvT, CvTConfig, CvTStageConfig, DINO,
    DINOConfig, DeepViT, Distill, DistillConfig, ImageSize, LeViT, LeViTConfig, LocalViT, MAE,
    MAEConfig, MP3, MP3Config, MPP, MPPConfig, MaxViT, MaxViTConfig, MobileViT, MobileViTConfig,
    NaViT, NaViTConfig, NaViTNestedTensor, NaViTNestedTensorConfig, NesT, NesTConfig, ParallelViT,
    PiT, PiTConfig, Pool, RegionViT, RegionViTConfig, SepViT, SepViTConfig, SimMIM, SimMIMConfig,
    SimpleViT, SimpleViT1D, SimpleViT3D, SimpleViTWithPatchDropout, SimpleViTWithQkNorm,
    SimpleViTWithRegisterTokens, TwinsSVT, TwinsSVTConfig, TwinsSVTStageConfig, VAAT, VAATConfig,
    VAT, VATConfig, ViT, ViT1D, ViT1DConfig, ViT3D, ViT3DConfig, ViTConfig, ViTForSmallDataset,
    ViTWithDecorr, ViTWithDecorrConfig, ViTWithPatchDropout, ViViT, ViViTConfig, XCiT, XCiTConfig,
};

#[test]
fn vit_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(256),
            patch_size: ImageSize::square(32),
            num_classes: 1000,
            dim: 1024,
            depth: 6,
            heads: 16,
            mlp_dim: 2048,
            dropout: 0.1,
            emb_dropout: 0.1,
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 256, 256], (Kind::Float, Device::Cpu));
    let preds = model.forward_t(&img, false);

    assert_eq!(preds.size(), [1, 1000]);
}

#[test]
fn vit_mean_pool_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(16),
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            mlp_dim: 256,
            pool: Pool::Mean,
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let preds = model.forward_t(&img, false);

    assert_eq!(preds.size(), [2, 10]);
}

#[test]
fn vit_rectangular_inputs_work() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::new(256, 128),
            patch_size: ImageSize::new(32, 16),
            num_classes: 1000,
            dim: 256,
            depth: 2,
            heads: 8,
            mlp_dim: 512,
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 256, 128], (Kind::Float, Device::Cpu));
    let preds = model.forward_t(&img, false);

    assert_eq!(preds.size(), [1, 1000]);
}

#[test]
fn simple_vit_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = SimpleViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(256),
            patch_size: ImageSize::square(32),
            num_classes: 1000,
            dim: 1024,
            depth: 6,
            heads: 16,
            mlp_dim: 2048,
            ..Default::default()
        },
    );
    let img = Tensor::randn([1, 3, 256, 256], (Kind::Float, Device::Cpu));
    let preds = model.forward_t(&img, false);

    assert_eq!(preds.size(), [1, 1000]);
}

#[test]
fn vit_with_decorr_outputs_logits_and_aux_loss() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViTWithDecorr::new(
        &vs.root(),
        ViTWithDecorrConfig {
            vit: ViTConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(4),
                num_classes: 100,
                dim: 128,
                depth: 2,
                heads: 8,
                dim_head: 16,
                mlp_dim: 512,
                ..Default::default()
            },
            decorr_sample_frac: 1.0,
        },
    );
    let img = Tensor::randn([2, 3, 32, 32], (Kind::Float, Device::Cpu));
    let (logits, aux_loss) = model.forward_t_with_aux(&img, true, true);

    assert_eq!(logits.size(), [2, 100]);
    assert_eq!(aux_loss.size(), [] as [i64; 0]);
}

#[test]
fn vit_1d_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViT1D::new(
        &vs.root(),
        ViT1DConfig {
            seq_len: 256,
            patch_size: 16,
            num_classes: 1000,
            dim: 256,
            depth: 2,
            heads: 8,
            mlp_dim: 512,
            ..Default::default()
        },
    );
    let series = Tensor::randn([4, 3, 256], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&series, false);

    assert_eq!(logits.size(), [4, 1000]);
}

#[test]
fn simple_vit_1d_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = SimpleViT1D::new(
        &vs.root(),
        ViT1DConfig {
            seq_len: 256,
            patch_size: 16,
            num_classes: 1000,
            dim: 256,
            depth: 2,
            heads: 8,
            mlp_dim: 512,
            ..Default::default()
        },
    );
    let series = Tensor::randn([4, 3, 256], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&series, false);

    assert_eq!(logits.size(), [4, 1000]);
}

#[test]
fn vit_3d_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViT3D::new(
        &vs.root(),
        ViT3DConfig {
            image_size: ImageSize::square(64),
            image_patch_size: ImageSize::square(16),
            frames: 8,
            frame_patch_size: 2,
            num_classes: 1000,
            dim: 256,
            depth: 2,
            heads: 8,
            mlp_dim: 512,
            ..Default::default()
        },
    );
    let video = Tensor::randn([2, 3, 8, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&video, false);

    assert_eq!(logits.size(), [2, 1000]);
}

#[test]
fn simple_vit_3d_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = SimpleViT3D::new(
        &vs.root(),
        ViT3DConfig {
            image_size: ImageSize::square(64),
            image_patch_size: ImageSize::square(16),
            frames: 8,
            frame_patch_size: 2,
            num_classes: 1000,
            dim: 256,
            depth: 2,
            heads: 8,
            mlp_dim: 512,
            ..Default::default()
        },
    );
    let video = Tensor::randn([2, 3, 8, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&video, false);

    assert_eq!(logits.size(), [2, 1000]);
}

#[test]
fn simple_vit_with_register_tokens_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = SimpleViTWithRegisterTokens::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(16),
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            mlp_dim: 256,
            ..Default::default()
        },
        4,
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn simple_vit_with_patch_dropout_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = SimpleViTWithPatchDropout::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(16),
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            mlp_dim: 256,
            ..Default::default()
        },
        0.5,
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, true);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn vit_with_patch_dropout_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViTWithPatchDropout::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(16),
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            mlp_dim: 256,
            ..Default::default()
        },
        0.25,
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, true);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn simple_vit_with_qk_norm_outputs_embedding_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = SimpleViTWithQkNorm::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(16),
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            dim_head: 32,
            mlp_dim: 256,
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let embeddings = model.forward_t(&img, false);

    assert_eq!(embeddings.size(), [2, 128]);
}

#[test]
fn deep_vit_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = DeepViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(16),
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            dim_head: 32,
            mlp_dim: 256,
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn parallel_vit_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ParallelViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(16),
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            dim_head: 32,
            mlp_dim: 256,
            ..Default::default()
        },
        2,
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn local_vit_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = LocalViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(16),
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            dim_head: 32,
            mlp_dim: 256,
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn cct_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = CCT::new(
        &vs.root(),
        CCTConfig {
            image_size: ImageSize::square(64),
            num_classes: 10,
            embedding_dim: 128,
            num_layers: 2,
            num_heads: 4,
            mlp_ratio: 2,
            channels: 3,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            pooling_kernel_size: 3,
            pooling_stride: 2,
            pooling_padding: 1,
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn vit_for_small_dataset_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = ViTForSmallDataset::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::square(64),
            patch_size: ImageSize::square(8),
            num_classes: 10,
            dim: 128,
            depth: 2,
            heads: 4,
            dim_head: 32,
            mlp_dim: 256,
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn cvt_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = CvT::new(
        &vs.root(),
        CvTConfig {
            num_classes: 10,
            stages: [
                CvTStageConfig::new(32, 3, 2, 3, 1, 1, 1, 2),
                CvTStageConfig::new(64, 3, 2, 3, 1, 2, 1, 2),
                CvTStageConfig::new(128, 3, 2, 3, 1, 4, 1, 2),
            ],
            dim_head: 32,
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn pit_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = PiT::new(
        &vs.root(),
        PiTConfig {
            image_size: 64,
            patch_size: 8,
            num_classes: 10,
            dim: 64,
            depths: vec![1, 1],
            heads: vec![2, 4],
            mlp_dim: 128,
            dim_head: 32,
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn levit_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = LeViT::new(
        &vs.root(),
        LeViTConfig {
            image_size: 64,
            num_classes: 10,
            dims: vec![64, 96],
            depths: vec![1, 1],
            heads: vec![2, 3],
            mlp_mult: 2,
            stages: 2,
            dim_key: 16,
            dim_value: 32,
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn levit_outputs_distill_logits_when_configured() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = LeViT::new(
        &vs.root(),
        LeViTConfig {
            image_size: 64,
            num_classes: 10,
            dims: vec![64, 96],
            depths: vec![1, 1],
            heads: vec![2, 3],
            mlp_mult: 2,
            stages: 2,
            dim_key: 16,
            dim_value: 32,
            num_distill_classes: Some(5),
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let (logits, distill) = model.forward_t_with_distill(&img, false);

    assert_eq!(logits.size(), [2, 10]);
    assert_eq!(distill.expect("distill logits").size(), [2, 5]);
}

fn compact_test_config(num_classes: i64) -> CompactVisionConfig {
    CompactVisionConfig {
        image_size: ImageSize::square(32),
        patch_size: ImageSize::square(8),
        num_classes,
        dim: 64,
        depth: 1,
        heads: 2,
        mlp_dim: 128,
        dim_head: 32,
        ..Default::default()
    }
}

#[test]
fn mobile_vit_outputs_logits_shape() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = MobileViT::new(
        &vs.root(),
        MobileViTConfig {
            image_size: ImageSize::square(64),
            num_classes: 10,
            dims: [32, 48, 64],
            channels: [8, 8, 16, 16, 24, 24, 32, 32, 48, 48, 96],
            depths: [1, 1, 1],
            ..Default::default()
        },
    );
    let img = Tensor::randn([2, 3, 64, 64], (Kind::Float, Device::Cpu));
    let logits = model.forward_t(&img, false);

    assert_eq!(logits.size(), [2, 10]);
}

#[test]
fn attention_family_models_output_logits_shape() {
    let img = Tensor::randn([2, 3, 32, 32], (Kind::Float, Device::Cpu));

    let vs = nn::VarStore::new(Device::Cpu);
    assert_eq!(
        CaiT::new(
            &vs.root(),
            CaiTConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(8),
                num_classes: 10,
                dim: 64,
                depth: 1,
                cls_depth: 1,
                heads: 2,
                dim_head: 32,
                mlp_dim: 128,
                ..Default::default()
            },
        )
        .forward_t(&img, false)
        .size(),
        [2, 10]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    assert_eq!(
        XCiT::new(
            &vs.root(),
            XCiTConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(8),
                num_classes: 10,
                dim: 64,
                depth: 1,
                cls_depth: 1,
                heads: 2,
                dim_head: 32,
                mlp_dim: 128,
                ..Default::default()
            },
        )
        .forward_t(&img, false)
        .size(),
        [2, 10]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    assert_eq!(
        NesT::new(
            &vs.root(),
            NesTConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(4),
                num_classes: 10,
                dim: 32,
                heads: 2,
                num_hierarchies: 2,
                block_repeats: vec![1, 1],
                mlp_mult: 2,
                ..Default::default()
            },
        )
        .forward_t(&img, false)
        .size(),
        [2, 10]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    assert_eq!(
        SepViT::new(
            &vs.root(),
            SepViTConfig {
                ..compact_test_config(10)
            },
        )
        .forward_t(&img, false)
        .size(),
        [2, 10]
    );
}

#[test]
fn hierarchical_attention_family_models_output_logits_shape() {
    let img = Tensor::randn([2, 3, 32, 32], (Kind::Float, Device::Cpu));
    let max_vit_img = Tensor::randn([2, 3, 112, 112], (Kind::Float, Device::Cpu));

    let vs = nn::VarStore::new(Device::Cpu);
    assert_eq!(
        CrossFormer::new(
            &vs.root(),
            CrossFormerConfig {
                ..compact_test_config(10)
            },
        )
        .forward_t(&img, false)
        .size(),
        [2, 10]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    assert_eq!(
        TwinsSVT::new(
            &vs.root(),
            TwinsSVTConfig {
                num_classes: 10,
                stages: [
                    TwinsSVTStageConfig::new(16, 2, 2, 2, 1),
                    TwinsSVTStageConfig::new(24, 2, 2, 2, 1),
                    TwinsSVTStageConfig::new(32, 2, 2, 2, 1),
                    TwinsSVTStageConfig::new(48, 2, 2, 1, 1),
                ],
                heads: 2,
                dim_head: 8,
                mlp_mult: 2,
                ..Default::default()
            },
        )
        .forward_t(&img, false)
        .size(),
        [2, 10]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    assert_eq!(
        MaxViT::new(
            &vs.root(),
            MaxViTConfig {
                num_classes: 10,
                dim: 16,
                depth: vec![1, 1],
                dim_head: 8,
                dim_conv_stem: Some(16),
                window_size: 7,
                mbconv_expansion_rate: 2,
                ..Default::default()
            },
        )
        .forward_t(&max_vit_img, false)
        .size(),
        [2, 10]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    assert_eq!(
        RegionViT::new(
            &vs.root(),
            RegionViTConfig {
                ..compact_test_config(10)
            },
        )
        .forward_t(&img, false)
        .size(),
        [2, 10]
    );
}

#[test]
fn dynamic_and_nested_models_output_logits_shape() {
    let img = Tensor::randn([2, 3, 32, 32], (Kind::Float, Device::Cpu));
    let mask = Tensor::ones([2, 16], (Kind::Bool, Device::Cpu));

    let vs = nn::VarStore::new(Device::Cpu);
    let navit = NaViT::new(
        &vs.root(),
        NaViTConfig {
            ..compact_test_config(10)
        },
    );
    assert_eq!(
        navit.forward_t_with_mask(&img, &mask, false).size(),
        [2, 10]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    let nested = NaViTNestedTensor::new(
        &vs.root(),
        NaViTNestedTensorConfig {
            ..compact_test_config(10)
        },
    );
    assert_eq!(
        nested
            .forward_nested_t(&[img.get(0), img.get(1)], false)
            .size(),
        [2, 10]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    let ats = ATS::new(
        &vs.root(),
        ATSConfig {
            ..compact_test_config(10)
        },
    );
    assert_eq!(
        ats.forward_t_with_keep_ratio(&img, 0.5, false).size(),
        [2, 10]
    );
}

#[test]
fn training_wrappers_output_scalar_losses() {
    let img = Tensor::randn([2, 3, 32, 32], (Kind::Float, Device::Cpu));
    let target = Tensor::randn([2, 10], (Kind::Float, Device::Cpu));

    let vs = nn::VarStore::new(Device::Cpu);
    let mpp = MPP::new(
        &vs.root(),
        MPPConfig {
            ..compact_test_config(10)
        },
    );
    assert_eq!(
        mpp.forward_t_with_target(&img, &target, true).size(),
        [] as [i64; 0]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    let mp3 = MP3::new(
        &vs.root(),
        MP3Config {
            ..compact_test_config(10)
        },
    );
    assert_eq!(
        mp3.forward_t_with_target(&img, &target, true).size(),
        [] as [i64; 0]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    let dino = DINO::new(
        &vs.root(),
        DINOConfig {
            ..compact_test_config(10)
        },
    );
    assert_eq!(
        dino.forward_t_with_teacher(&img, &target, true).size(),
        [] as [i64; 0]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    let distill = Distill::new(
        &vs.root(),
        DistillConfig {
            student: compact_test_config(10),
            teacher_dim: 10,
            distill_dim: 10,
            temperature: 1.0,
        },
    );
    assert_eq!(
        distill.forward_t_with_teacher(&img, &target, true).size(),
        [] as [i64; 0]
    );
}

#[test]
fn video_audio_multimodal_models_output_logits_shape() {
    let video = Tensor::randn([2, 3, 4, 32, 32], (Kind::Float, Device::Cpu));

    let vs = nn::VarStore::new(Device::Cpu);
    let vivit = ViViT::new(
        &vs.root(),
        ViViTConfig {
            image_size: ImageSize::square(32),
            image_patch_size: ImageSize::square(8),
            frames: 4,
            frame_patch_size: 2,
            num_classes: 10,
            dim: 64,
            depth: 1,
            heads: 2,
            mlp_dim: 128,
            dim_head: 32,
            ..Default::default()
        },
    );
    assert_eq!(vivit.forward_t(&video, false).size(), [2, 10]);

    let vs = nn::VarStore::new(Device::Cpu);
    let wrapper = AcceptVideoWrapper::new(
        &vs.root(),
        AcceptVideoWrapperConfig {
            image_model: compact_test_config(10),
            frames: 4,
        },
    );
    assert_eq!(wrapper.forward_t(&video, false).size(), [2, 10]);

    let vs = nn::VarStore::new(Device::Cpu);
    let vat = VAT::new(
        &vs.root(),
        VATConfig {
            image_size: ImageSize::square(32),
            patch_size: ImageSize::square(8),
            frames: 4,
            action_dim: 6,
            num_classes: 10,
            dim: 64,
            depth: 1,
            heads: 2,
            mlp_dim: 128,
            dim_head: 32,
            ..Default::default()
        },
    );
    let actions = Tensor::randn([2, 4, 6], (Kind::Float, Device::Cpu));
    assert_eq!(
        vat.forward_t_with_actions(&video, &actions, false).size(),
        [2, 10]
    );

    let vs = nn::VarStore::new(Device::Cpu);
    let vaat = VAAT::new(
        &vs.root(),
        VAATConfig {
            vat: VATConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(8),
                frames: 4,
                action_dim: 6,
                num_classes: 10,
                dim: 64,
                depth: 1,
                heads: 2,
                mlp_dim: 128,
                dim_head: 32,
                ..Default::default()
            },
            audio_bins: 8,
            audio_frames: 8,
        },
    );
    let audio = Tensor::randn([2, 8, 8], (Kind::Float, Device::Cpu));
    assert_eq!(
        vaat.forward_t_with_audio_actions(&video, &audio, &actions, false)
            .size(),
        [2, 10]
    );
}

#[test]
fn mae_outputs_scalar_loss() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = MAE::new(
        &vs.root(),
        MAEConfig {
            encoder: ViTConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(8),
                num_classes: 0,
                dim: 128,
                depth: 2,
                heads: 4,
                dim_head: 32,
                mlp_dim: 256,
                ..Default::default()
            },
            decoder_dim: 64,
            masking_ratio: 0.5,
            decoder_depth: 1,
            decoder_heads: 4,
            decoder_dim_head: 16,
        },
    );
    let img = Tensor::randn([2, 3, 32, 32], (Kind::Float, Device::Cpu));
    let loss = model.forward_t(&img, true);

    assert_eq!(loss.size(), [] as [i64; 0]);
}

#[test]
fn simmim_outputs_scalar_loss() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = SimMIM::new(
        &vs.root(),
        SimMIMConfig {
            encoder: ViTConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(8),
                num_classes: 0,
                dim: 128,
                depth: 2,
                heads: 4,
                dim_head: 32,
                mlp_dim: 256,
                ..Default::default()
            },
            masking_ratio: 0.5,
        },
    );
    let img = Tensor::randn([2, 3, 32, 32], (Kind::Float, Device::Cpu));
    let loss = model.forward_t(&img, true);

    assert_eq!(loss.size(), [] as [i64; 0]);
}

#[test]
#[should_panic(expected = "masking ratio must be between 0 and 1")]
fn mae_rejects_zero_masking_ratio() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = MAE::new(
        &vs.root(),
        MAEConfig {
            encoder: ViTConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(8),
                num_classes: 0,
                dim: 128,
                depth: 2,
                heads: 4,
                dim_head: 32,
                mlp_dim: 256,
                ..Default::default()
            },
            decoder_dim: 64,
            masking_ratio: 0.0,
            decoder_depth: 1,
            decoder_heads: 4,
            decoder_dim_head: 16,
        },
    );
    let img = Tensor::randn([2, 3, 32, 32], (Kind::Float, Device::Cpu));

    let _ = model.forward_t(&img, true);
}

#[test]
#[should_panic(expected = "masking ratio must be between 0 and 1")]
fn simmim_rejects_full_masking_ratio() {
    let vs = nn::VarStore::new(Device::Cpu);
    let model = SimMIM::new(
        &vs.root(),
        SimMIMConfig {
            encoder: ViTConfig {
                image_size: ImageSize::square(32),
                patch_size: ImageSize::square(8),
                num_classes: 0,
                dim: 128,
                depth: 2,
                heads: 4,
                dim_head: 32,
                mlp_dim: 256,
                ..Default::default()
            },
            masking_ratio: 1.0,
        },
    );
    let img = Tensor::randn([2, 3, 32, 32], (Kind::Float, Device::Cpu));

    let _ = model.forward_t(&img, true);
}

#[test]
#[should_panic(expected = "image dimensions must be divisible by patch size")]
fn vit_rejects_invalid_patch_size() {
    let vs = nn::VarStore::new(Device::Cpu);
    let _ = ViT::new(
        &vs.root(),
        ViTConfig {
            image_size: ImageSize::new(255, 256),
            patch_size: ImageSize::square(32),
            ..Default::default()
        },
    );
}
