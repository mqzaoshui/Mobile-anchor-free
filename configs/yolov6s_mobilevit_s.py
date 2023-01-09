# YOLOv6s model with mobilevit_s backbone
model = dict(
    type='YOLOv6s',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.50,
    backbone=dict(
        type='MobileViT',
        size='sss',
        img_size=(256, 256),
        dims=[144, 192, 240],
        channels=[16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 128, 256, 512], 
        expansion=2,
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        ),
    neck=dict(
        type='RepPANNeck',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=1,
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        iou_type='giou',
        use_dfl=False,
        reg_max=0 #if use_dfl is False, please set reg_max to 0
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)

# preprocessor data for vit
eval_params = dict(
        img_size=640,
        test_load_size=640,
        batch_size=12,
        letterbox_return_int=True,
        scale_exact=True,
        force_no_pad=True,
        not_infer_on_rect=True,
    )
