_base_ = [
    '../_base_/cleansed_xraydata2.py',
    '../_base_/default_runtime.py', '../_base_/schedule_160k.py'
]

# model settings
num_classes = 29
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (769, 769)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoderWithoutArgmax',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(2, 2),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(2, 2, 2),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(2, 2, 2, 2),
                num_channels=(18, 36, 72, 144)))),
    decode_head=dict(
        type='UPerHeadWithoutAccuracy',
        in_channels=[18, 36, 72, 144],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=sum([18, 36, 72, 144]),
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        threshold=0.5,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=True, loss_weight=3.0)
        ]
        ),
    auxiliary_head=dict(
        type='FCNHeadWithoutAccuracy',
        in_channels=[18, 36, 72, 144],
        in_index=(0, 1, 2, 3),
        channels=sum([18, 36, 72, 144]),
        input_transform='resize_concat',
        kernel_size=1,
        num_convs=1,
        concat_input=False,
        dropout_ratio=-1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4),
            dict(type='DiceLoss', use_sigmoid=True, loss_weight=1.0)
        ]
        ),
    # model training and testing settings
    
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


train_dataloader = dict(batch_size = 2)