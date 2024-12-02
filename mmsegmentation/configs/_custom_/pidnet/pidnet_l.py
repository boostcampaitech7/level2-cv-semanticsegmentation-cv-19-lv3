_base_ = [
    '../_base_/xraydata2.py',
    '../_base_/default_runtime.py', '../_base_/schedule_160k.py'
]

crop_size = (1024, 1024)
num_classes = 29
load_from = "https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes/pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth"
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-l_imagenet1k_20230306-67889109.pth'  # noqa
class_weight = [
    0.81, 0.80, 0.80, 0.80, 0.80, 
    0.80, 0.80, 0.80, 0.80, 0.80, 
    0.80, 0.80, 0.80, 0.80, 0.80, 
    0.81, 0.81, 0.80, 0.80, 0.82, 
    0.88, 0.81, 0.82, 0.81, 0.82, 
    0.82, 0.92, 0.80, 0.80
]


data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=64,
        ppm_channels=112,
        num_stem_blocks=3,
        num_branch_blocks=4,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='PIDHead',
        in_channels=256,
        channels=256,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=class_weight,
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


IMAGE_SIZE = (1480, 1480)
train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadXRayAnnotations'),
            dict(type='Resize', scale=IMAGE_SIZE),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='TransposeAnnotations'),
            dict(type='GenerateEdge', edge_width=4),
            dict(type='PackSegInputs')
        ]
train_dataloader = dict(batch_size=1, dataset=dict(pipeline=train_pipeline))


iters = 30000
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]
# training schedule for 120k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')