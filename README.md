# 类别选择
PIDRAY
基类：Baton Pliers Hammer Wrench 警棍 钳子 锤子 扳手
新类：handCuffs 手铐

可以直接通过metainfo 选择数据集中部分类别进行训练，mmdection会自动根据metainfo指定的类别进行过滤。

# 小样本设定
/home/xray/xray/mmdetection/mmdet/datasets/coco.py
修改coco数据集，filter_data部分，增加filter_cfg字典的K_shot键，在数据集config文件 /home/xray/xray/mmdetection/configs/_base_/datasets/pidray_dataset_4+1.py 中的train_dataloader传入filter_cfg
```python
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
        filter_cfg=dict(filter_empty_gt=True, min_size=32, K_shot=10),
        pipeline=train_pipeline_incremental,
        backend_args=backend_args))  

```
在coco filter_data 函数中添加样本数量限定的逻辑代码
```python
if K_shot == -1:
            # 源码部分
            valid_data_infos = []
            for i, data_info in enumerate(self.data_list):
                img_id = data_info['img_id']
                width = data_info['width']
                height = data_info['height']
                if filter_empty_gt and img_id not in ids_in_cat:
                    continue
                if min(width, height) >= min_size:
                    valid_data_infos.append(data_info)
        else:
            # 初始化每个类被的样本数量
            category_sample_count = {cat_id: 0 for cat_id in self.cat_ids}
            valid_data_infos = []
            for i, data_info in enumerate(self.data_list):
                img_id = data_info['img_id']
                width = data_info['width']
                height = data_info['height']
                if filter_empty_gt and img_id not in ids_in_cat:
                    continue
                if min(width, height) < min_size:
                    continue
                keep_image = False
                # 检查图片包含的类别
                for instance in data_info['instances']:
                    # print(f'instance {instance}')
                    category_id = self.cat_ids[instance['bbox_label']]
                    if category_sample_count[category_id] < K_shot:
                        category_sample_count[category_id] += 1
                        print(f'category_sample_count{category_sample_count}')
                        keep_image = True
                        
                if keep_image:
                    print(f'保留图片')
                    valid_data_infos.append(data_info)
                    input()
                # else:
                    # print(f'超出 10 shot')
                # input()
        print(f'valid_data_infos {valid_data_infos}')
        return valid_data_infos
```

# 基类训练
使用配置文件 /home/xray/xray/mmdetection/configs/cascade_rcnn/pidray_base4_cascade_mask_rcnn_r101_with_R0R1.py 
设定其中的num_classes = 4 
数据集配置文件 /home/xray/xray/mmdetection/configs/_base_/datasets/pidray_dataset_4+1.py
设定dataload为incremental对应的 位于文件最后
运行命令
```
CUDA_VISIBLE_DEVICES=2,3,4,5 bash ./tools/dist_train.sh configs/cascade_rcnn/pidray_incremental1_cascade_mask_rcnn_r101_with_R0R1.py  4
CUDA_VISIBLE_DEVICES=2,3,4,5 bash tools/dist_train.sh configs/cascade_rcnn/pidray_base4_cascade_mask_rcnn_r101.py 4 --auto-scale-lr
```

# 扩展分类维度
使用/home/xray/xray/mmdetection/utils/change_checkpoints.py 调整最后的bbox head和mask head 分类头维度，注意bbox head含有背景类，为实际类别+1，背景类是最后一维。

例：
原bbox head fc_cls weight维度为(5,1024),对应4类基类和1类背景类，在进行维度扩展后，为(6,1024), 对应4类基类+1类新类+1类背景类。   
在进行权重参数赋值时，需要注意将原参数的前4维赋值到新参数的前四维，最后一维（背景）复制到新参数的最后一维，空出新参数的倒数第2维对应新增类别的分类。  
若简单的将旧参数赋值到新参数的前5维，会导致错位，原背景类参数对应到新类别的参数，导致测试结果大部分分到了背景类和新类，基类几乎无预测结果。 从混淆矩阵结果可以验证。

不错位的结果：
![Alt text](cascade_pidray_incremental1/confusion_matrix.png)

错位结果：  
混淆矩阵右侧背景类预测很多，前四类基类几乎无预测。
## 增量训练

/home/xray/xray/mmdetection/mmdet/engine/hooks/freeze_part_param_hook.py 是否需要冻结背景类
/home/xray/xray/mmdetection/configs/_base_/schedules/pid_schedule.py 增量过程训练epoch设为100，设置最大保存模型数量

小样本训练不适合开多卡，每个卡上batch size较小，导致不稳定。

运行命令
```
python tools/train.py configs/cascade_rcnn/pidray_incremental1_cascade_mask_rcnn_r101.py --auto-scale-lr
```

# 测试

测试
```
CUDA_VISIBLE_DEVICES=2,3,4,5 bash ./tools/dist_test.sh configs/cascade_rcnn/pidray_incremental1_cascade_mask_rcnn_r101_with_R0R1.py ./workdir/cascade_pidray_base4/epoch_12_modified.pth 4 --out ./workdir/cascade_pidray_incremental1
```
保存pkl文件用于混淆矩阵分析

混淆矩阵分析
```
python tools/analysis_tools/confusion_matrix.py configs/cascade_rcnn/pidray_incremental1_cascade_mask_rcnn_r101_with_R0R1.py  ./workdir/cascade_pidray_incremental1/result.pkl  ./workdir/cascade_pidray_incremental1
```