from typing import Optional
from mmengine.model import is_model_wrapper
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS

@HOOKS.register_module()
class FreezeHeadPartsHook(Hook):
    """Freeze Parts of ROI Head Layers Hook.

    This hook freezes parts of the bbox head and mask head layers during training.

    Args:
        interval (int): How often to reset the parameters (every k iterations).
            Default: 1 (every iteration).
    """

    def __init__(self, interval: int = 1, base_classes_num: int = 4) -> None:
        self.interval = interval
        self.base_classes_num = base_classes_num
        self.initial_weights = {}
        self.initial_biases = {}

    def before_train_epoch(self, runner: Runner) -> None:
        """Save the initial weights and biases of the relevant layers before training."""
        # 保存 bbox head 和 mask head 的初始权重和偏置
        for i in range(3):  # Assuming 3 stages for bbox and mask heads
            # BBox Head
            # if hasattr(runner.model.module, 'roi_head'):
                # print('roi head in model')
                # input()
            # print(runner.model.modules)
            model = runner.model
            if is_model_wrapper(model):
                model = model.module
            # input('check runner')
            bbox_head = model.roi_head.bbox_head[i]
            # print(bbox_head.fc_cls.weight)
            # input('before train epoch bbox head param')
            # TODO: 背景类是否需要冻结？？
            self.initial_weights[f'bbox_head_{i}_weight'] = bbox_head.fc_cls.weight[:self.base_classes_num, :].clone().detach()
            self.initial_biases[f'bbox_head_{i}_bias'] = bbox_head.fc_cls.bias[:self.base_classes_num].clone().detach()

            # Mask Head
            mask_head = model.roi_head.mask_head[i]
            self.initial_weights[f'mask_head_{i}_weight'] = mask_head.conv_logits.weight[:self.base_classes_num, :, :, :].clone().detach()
            self.initial_biases[f'mask_head_{i}_bias'] = mask_head.conv_logits.bias[:self.base_classes_num].clone().detach()

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Restore the relevant weights and biases after each training iteration."""
        if self.every_n_train_iters(runner, self.interval):
            for i in range(3):  # Assuming 3 stages for bbox and mask heads
                # BBox Head
                model = runner.model
                if is_model_wrapper(model):
                    model = model.module
                bbox_head = model.roi_head.bbox_head[i]
                with torch.no_grad():
                    bbox_head.fc_cls.weight[:self.base_classes_num, :] = self.initial_weights[f'bbox_head_{i}_weight']
                    bbox_head.fc_cls.bias[:self.base_classes_num] = self.initial_biases[f'bbox_head_{i}_bias']

                # Mask Head
                mask_head = model.roi_head.mask_head[i]
                with torch.no_grad():
                    mask_head.conv_logits.weight[:self.base_classes_num, :, :, :] = self.initial_weights[f'mask_head_{i}_weight']
                    mask_head.conv_logits.bias[:self.base_classes_num] = self.initial_biases[f'mask_head_{i}_bias']
            # print(runner.model.module.roi_head.bbox_head[0].fc_cls.weight)
            # input('after train iter bbox head param')