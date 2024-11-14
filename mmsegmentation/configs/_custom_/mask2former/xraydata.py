resize_size = (1748, 1748)
crop_size = (1536, 1536)
test_size = (2048, 2048)

dataset_type = 'XRayDataset'
data_root = '/data/ephemeral/home/cityscapes_format_xlay/'


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(
    #     type='RandomChoiceResize',
    #     scales=[int(1024 * x * 0.1) for x in range(5, 21)],
    #     resize_type='ResizeShortestEdge',
    #     max_size=4096),
    dict(type='Resize', scale=resize_size, keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        img_suffix='.png',
        seg_map_suffix='_gtFine_labelIds.png',
        pipeline=train_pipeline))

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=test_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=test_size, keep_ratio=True),
    dict(type='PackSegInputs')
]

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        img_suffix='.png',
        seg_map_suffix='_gtFine_labelIds.png',
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='leftImg8bit/test'),
        img_suffix='.png',
        pipeline=test_pipeline))


val_evaluator = dict(type='DiceMetric')
test_evaluator = dict(type='SubmissionMetric', save_path = "/data/ephemeral/home/submission")
# val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
# test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='PackSegInputs')]
        ])
]