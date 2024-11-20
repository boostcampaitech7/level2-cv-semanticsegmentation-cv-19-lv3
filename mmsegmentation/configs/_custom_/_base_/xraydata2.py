# dataset settings
dataset_type = 'XRayDataset2'


train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadXRayAnnotations'),
            dict(type='Resize', scale=(1280, 1280)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RandomRotate', prob=0.5, degree=15),
            dict(type='PhotoMetricDistortion'),
            dict(type='TransposeAnnotations'),
            dict(type='PackSegInputs')
        ]
val_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadXRayAnnotations'),
            dict(type='TransposeAnnotations'),
            dict(type='PackSegInputs')
        ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(type='PackSegInputs')
]


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
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    # persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        is_train = True,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        is_train = False,
        pipeline=val_pipeline))


test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='XRayDataset',
        data_root='/data/ephemeral/home/cityscapes_format_xlay_kfold',
        data_prefix=dict(img_path='leftImg8bit/test'),
        img_suffix='.png',
        pipeline=test_pipeline))


val_evaluator = dict(type='DiceMetricForMultiLabel')
test_evaluator = dict(type='SubmissionMetric')