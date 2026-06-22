#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import contextlib

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, moco=None, freeze=False):  #将freeze_detector设置为True，这样在训练模式下，模型会冻结检测器部分，只训练ReID头,需要取消注释下面的两行代码
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()      # backbone, CSPNet with PANet
        if head is None:
            head = YOLOXHead(80)        # head

        self.backbone = backbone
        self.head = head
        self.moco = moco
        self.freeze_detector = freeze
    
    # override
    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.freeze_detector and mode is True:
            # train reid only   
            self.backbone.eval()        # 只想训练ReID头，而冻结其他的网络部分，那么这些行代码确实不需要注释掉。
            self.head.stems.eval()      # 相反，它们是必须保留的，因为这些代码会将你不希望参与训练的部分（backbone和检测头）设置为评估模式（eval模式），
            self.head.cls_convs.eval()  # 从而确保这些部分的参数在训练过程中不会更新。这样一来，模型的backbone和检测头将被冻结，而只训练ReID头的参数。
            self.head.reg_convs.eval()
            self.head.cls_preds.eval()
            self.head.reg_preds.eval()
            self.head.obj_preds.eval()
            # self.head.reid_convs.train()
            # self.head.reid_preds.train()

        return self

    def forward(self, x, targets=None, infos=None):
        if len(x.shape) > 4:  # stacked auxiliary images
            x, x_aug, x_prev = x.unbind(dim=1)

        # fpn output content features of [dark3, dark4, dark5]
        ctx = torch.no_grad() if self.freeze_detector else contextlib.nullcontext()
        with ctx:
            fpn_outs = self.backbone(x)

        if self.training:
            targets, targets_aug, targets_prev = targets.unbind(dim=1)
            assert targets is not None
            assert self.moco is not None

            loss, iou_loss, conf_loss, cls_loss, id_loss, id_aux_loss, l1_loss, num_fg = self.head(       # TODO: ReID, update 'id_loss'
                fpn_outs, targets, x, context=[self.moco, self.backbone, infos, x_aug, x_prev, targets_aug, targets_prev], freeze_detector=self.freeze_detector
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "id_loss": id_loss,             # TODO: ReID, update 'id_loss'
                "id_aux_loss": id_aux_loss,             # TODO: ReID, update 'id_aux_loss'
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs