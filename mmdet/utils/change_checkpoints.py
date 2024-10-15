import torch
import torch.nn.init as init

# 加载检查点
pth_path = '/home/xray/xray/mmdetection/workdir/pidray_base4/epoch_12.pth'
checkpoint = torch.load(pth_path)
old_class_num = 4
new_calss_num = 5
# 提取状态字典
state_dict = checkpoint.get('state_dict', checkpoint)

# 修改状态字典中的参数
modified_state_dict = {}
for key, value in state_dict.items():
    if 'bbox_head' in key and 'fc_cls' in key:
        # 处理 bbox_head.fc_cls 参数
        old_shape = value.shape
        if 'weight' in key:
            new_shape = (new_calss_num + 1, 1024)  # 新的权重维度
            if old_shape != new_shape:
                print(f"Adjusting {key} from shape {old_shape} to {new_shape}")
                new_value = torch.zeros(new_shape)
                new_value[:old_shape[0] - 1, :] = value[:old_shape[0] - 1, :]  # 保留旧类别参数 不包括背景类
                new_value[-1, :] = value[-1, :] # 背景类参数
                # init.xavier_uniform_(new_value[old_shape[0] - 1, :])  # 对新增部分进行 Xavier 初始化
                modified_state_dict[key] = new_value
            else:
                modified_state_dict[key] = value

        elif 'bias' in key:
            new_shape = (new_calss_num + 1,)  # 新的 bias 维度是一维的
            if old_shape != new_shape:
                print(f"Adjusting {key} from shape {old_shape} to {new_shape}")
                new_value = torch.zeros(new_shape)
                new_value[:old_shape[0] - 1] = value[:old_shape[0] - 1]  # 保留旧类别参数 不包括背景类
                new_value[-1] = value[-1] # 背景类参数
                # init.xavier_uniform_(new_value[old_shape[0] - 1].unsqueeze(0))  # 对新增部分进行 Xavier 初始化
                modified_state_dict[key] = new_value
            else:
                modified_state_dict[key] = value

    elif 'mask_head' in key and 'conv_logits' in key:
        # 处理 mask_head.conv_logits 参数
        old_shape = value.shape
        if 'weight' in key:
            new_shape = (new_calss_num, 256, 1, 1)
            if old_shape != new_shape:
                print(f"Adjusting {key} from shape {old_shape} to {new_shape}")
                new_value = torch.zeros(new_shape)
                new_value[:old_shape[0], :, :, :] = value
                # init.xavier_uniform_(new_value[old_shape[0]:, :, :, :])  # 对新增部分进行 Xavier 初始化
                modified_state_dict[key] = new_value
            else:
                modified_state_dict[key] = value

        elif 'bias' in key:
            new_shape = (new_calss_num,)  # 新的 bias 维度是一维的
            if old_shape != new_shape:
                print(f"Adjusting {key} from shape {old_shape} to {new_shape}")
                new_value = torch.zeros(new_shape)
                new_value[:old_shape[0]] = value  # 保留旧参数
                # init.xavier_uniform_(new_value[old_shape[0]:].unsqueeze(0))  # 对新增部分进行 Xavier 初始化
                modified_state_dict[key] = new_value
            else:
                modified_state_dict[key] = value

    else:
        modified_state_dict[key] = value

# 保存修改后的状态字典
modified_pth_path = '/home/xray/xray/mmdetection/workdir/pidray_base4/epoch_12_modified.pth'
torch.save(modified_state_dict, modified_pth_path)

print(f"Modified checkpoint saved to {modified_pth_path}")
