IMAGE_SIZE = (1024, 1024)

# dataset settings
dataset_type = 'XRayDataset2'


train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadXRayAnnotations'),
            dict(type='Resize', scale=IMAGE_SIZE),
            dict(type='TransposeAnnotations'),
            dict(type='PackSegInputs')
        ]
val_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=IMAGE_SIZE),
            dict(type='LoadXRayAnnotations'),
            dict(type='TransposeAnnotations'),
            dict(type='PackSegInputs')
        ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=IMAGE_SIZE),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
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
    batch_size=8,
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
            ], 
            [
                dict(type='PackSegInputs')
            ]
        ])
]