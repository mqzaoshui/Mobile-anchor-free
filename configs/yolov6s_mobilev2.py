# YOLOv6s MobileNetV2 model
model = dict(
    type='YOLOv6s',
    pretrained=None,
    depth_multiple=0.33,
    width_multiple=0.50,
    backbone=dict(
        type='MobileNetV2',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        cfgs=[
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  32, 2, 2],
            [6,  64, 3, 1],
            [6, 128, 4, 2],
            [6, 256, 3, 2],
            [6, 512, 3, 2],
        ]
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
