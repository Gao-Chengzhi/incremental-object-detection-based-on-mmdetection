max_epoch = 100
# training schedule for 1x   val_interval base 1  incremental 10
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epoch, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
"""
# 基类训练
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
"""
# 小样本    增量训练
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=[50, 75, 90],
        gamma=0.1)
]

# optimizer
# origin lr=0.02 增量阶段，减半尝试
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02 / 2, momentum=0.9, weight_decay=0.0001),
    clip_grad=None,)

# 
# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
