IMAGE_SIZE = (2048, 2048)
IMAGE_ROOT = '/data/ephemeral/home/cleansed_data/train/DCM'
LABEL_ROOT = '/data/ephemeral/home/cleansed_data/train/outputs_json'
SUBMISSION_PATH = '/data/ephemeral/home/submission'

# dataset settings
dataset_type = 'XRayDataset2'


train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadXRayAnnotations'),
            dict(type='Resize', scale=IMAGE_SIZE),
            dict(
                type='Albu',  # Albumentations를 이용한 변환
                transforms=[
                    dict(type='HorizontalFlip', p=0.5),  # 수평 뒤집기
                    dict(type='ShiftScaleRotate', shift_limit=0.0, scale_limit=0.0, rotate_limit=15, p=0.5),  # 회전
                    dict(
                        type='ElasticTransform',  # 탄성 변환
                        alpha=10,
                        sigma=50,
                        p=0.5,
                    ),
                ]
            ),
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
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        is_train=True,
        image_root=IMAGE_ROOT,
        label_root=LABEL_ROOT,
        pipeline=train_pipeline
    )
    )
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        is_train=False,
        image_root=IMAGE_ROOT,
        label_root=LABEL_ROOT,
        pipeline=val_pipeline
    )
    )


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
test_evaluator = dict(type='SubmissionMetric', save_path=SUBMISSION_PATH, max_vis_cnt=20)



img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=IMAGE_SIZE),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(
                    type='Albu',  # Albumentations를 이용한 변환
                    transforms=[
                        dict(type='HorizontalFlip', p=0.5),  # 수평 뒤집기
                        dict(type='ShiftScaleRotate', shift_limit=0.0, scale_limit=0.0, rotate_limit=20, p=0.5),  # 회전
                        dict(
                            type='ElasticTransform',  # 탄성 변환
                            alpha=10,
                            sigma=100,
                            p=0.8,
                        ),
                    ],
                    keymap={'img': 'image'}
                ),
                # dict(
                #     type='Albu',  # Albumentations를 이용한 변환
                #     transforms=[
                #         dict(type='HorizontalFlip', p=0.5),  # 수평 뒤집기
                #         dict(type='ShiftScaleRotate', shift_limit=0.0, scale_limit=0.0, rotate_limit=20, p=0.5),  # 회전
                #         dict(
                #             type='ElasticTransform',  # 탄성 변환
                #             alpha=10,
                #             sigma=100,
                #             p=0.8,
                #         ),
                #     ],
                #     keymap={'img': 'image'}
                # ),
                # dict(
                #     type='Albu',  # Albumentations를 이용한 변환
                #     transforms=[
                #         dict(type='HorizontalFlip', p=0.5),  # 수평 뒤집기
                #         dict(type='ShiftScaleRotate', shift_limit=0.0, scale_limit=0.0, rotate_limit=20, p=0.5),  # 회전
                #         dict(
                #             type='ElasticTransform',  # 탄성 변환
                #             alpha=10,
                #             sigma=100,
                #             p=0.8,
                #         ),
                #     ],
                #     keymap={'img': 'image'}
                # ),
            ],
            [dict(type='PackSegInputs')]
        ])
]