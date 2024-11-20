val_save_interval = 1000
warm_up_step = 1000
max_iters = 30000
T_max = max_iters-warm_up_step

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    accumulative_counts=2,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
# learning policy
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=warm_up_step),
    dict(type='CosineAnnealingLR',
         T_max=T_max,
         by_epoch=False,
         begin=warm_up_step,
         end=max_iters)
]
optimizer_config = dict(type="GradientCumulativeOptimizerHook", cumulative_iters=2)

# training schedule for 90k
train_cfg = dict(type='IterBasedTrainLoop', 
                 max_iters=max_iters, 
                 val_interval=val_save_interval
                 )
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', 
                    by_epoch=False, 
                    interval=val_save_interval,
                    max_keep_ckpts=3,
                    save_best='mDice',
                    rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
    )