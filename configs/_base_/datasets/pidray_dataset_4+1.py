
## dataset 相关配置

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/xray/data/pidray/'
# pidray 全部类型
_all_metainfo = {
    'classes': ('Baton', 'Pliers', 'Hammer', 'Powerbank', 'Scissors', 'Wrench', 'Gun', 'Bullet', 'Sprayer', 'HandCuffs', 'Knife',
    'Lighter' ),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0)
               ]
}

all_metainfo = {
    'classes': ('Baton', 'Pliers', 'Hammer','Wrench', 'HandCuffs'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),(0, 0, 192)]
}
# 8个基类，前八个
base8_metainfo = {
    'classes': ('Baton', 'Pliers', 'Hammer', 'Powerbank', 'Scissors', 'Wrench', 'Gun', 'Bullet'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70)
               ]
}
# 4个基类：Baton Pliers Hammer Wrench 警棍 钳子 锤子 扳手 4个  标号：0，1，2，5
base_metainfo = {
    'classes': ('Baton', 'Pliers', 'Hammer', 'Wrench'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230)]
}
# 1个新类： handCuffs 手铐 标号 9
incremental_metainfo = {
    'classes': ('Baton', 'Pliers', 'Hammer', 'Wrench', 'HandCuffs'),
    'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),(0, 0, 192)]
}

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

# dict(type='LoadAnnotations_with_filter', with_bbox=True, with_mask=True, filter_classes=[0,1,2,3,4,5,6,7]),
train_pipeline_all = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(500, 500), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_pipeline_base = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(500, 500), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
train_pipeline_incremental = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # dict(type='LoadAnnotations_with_filter', with_bbox=True, with_mask=True, filter_classes=[9]),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(500, 500), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline_all = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(500, 500), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline_base = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(500, 500), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
# 构建训练数据集，基类训练与增量训练
train_dataloader_all = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=all_metainfo,
        ann_file='annotations/xray_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline_all,
        backend_args=backend_args))

train_dataloader_base = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=base8_metainfo,
        ann_file='annotations/xray_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline_base,
        backend_args=backend_args))  

# filter_cfg 添加小样本设定，每类选取10个样本 K SHOT K=10
train_dataloader_incremental = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=incremental_metainfo,
        ann_file='annotations/xray_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32, 
                            K_shot=10
                            ),
        pipeline=train_pipeline_incremental,
        backend_args=backend_args))  

val_dataloader_all = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=all_metainfo,
        ann_file='annotations/xray_test_easy.json',
        data_prefix=dict(img='easy/'),
        test_mode=True,
        pipeline=test_pipeline_all,
        backend_args=backend_args))

val_dataloader_base = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=base8_metainfo,
        ann_file='annotations/xray_test_easy.json',
        data_prefix=dict(img='easy/'),
        test_mode=True,
        pipeline=test_pipeline_base,
        backend_args=backend_args))

val_dataloader_incremental = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=incremental_metainfo,
        ann_file='annotations/xray_test_easy.json',
        data_prefix=dict(img='easy/'),
        test_mode=True,
        pipeline=test_pipeline_all,
        backend_args=backend_args))



test_dataloader_all = val_dataloader_all
test_dataloader_base = val_dataloader_base
test_dataloader_incremental = val_dataloader_incremental

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/xray_test_easy.json',
    metric=['bbox','segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator




# 使用关键词指定训练使用的数据集



# 基类 8 类 训练
"""
train_dataloader = train_dataloader_base
val_dataloader = val_dataloader_base
test_dataloader = test_dataloader_base

train_dataloader = train_dataloader_all
val_dataloader = val_dataloader_all
test_dataloader = test_dataloader_all

# 增量 1 类 训练
train_dataloader = train_dataloader_incremental
val_dataloader = val_dataloader_incremental
test_dataloader = test_dataloader_incremental
"""
train_dataloader = train_dataloader_base
val_dataloader = val_dataloader_base
test_dataloader = test_dataloader_base