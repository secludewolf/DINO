# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, clean_state_dict, is_main_process

from .position_encoding import build_position_encoding
from .convnext import build_convnext
from .swin_transformer import build_swin_transformer


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: List[int], return_interm_indices: list):
        super().__init__()
        # 是否从头训练backbone, 前几层提取的特征都大差不差, 一般没有必要重新训练
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update({"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)})

        # IntermediateLayerGetter这个类就是获取一个Model中你指定要获取的哪些层的输出，
        # 然后这些层的输出会在一个有序的字典中，字典中的key就是刚开始初始化这个类传进去的，value就是feature经过指定需要层的输出。
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}

        # return_layers = {'layer2': '1', 'layer3': '2', 'layer4': '3'}
        # 获取每一层的特征图
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # mask插值到与输出特征图尺寸一致
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # 注意BackboneBase的前向方法中的输入是NestedTensor这个类的实例，其实质就是将图像张量和对应的mask封装到一起
            out[name] = NestedTensor(x, mask)

        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 dilation: bool,
                 return_interm_indices: list,
                 batch_norm=FrozenBatchNorm2d,
                 ):
        assert name in ('resnet50', 'resnet101'), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]

        # 直接掉包 调用torchvision.models中的backbone
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=batch_norm)
        num_channels_all = [256, 512, 1024, 2048]
        num_channels = num_channels_all[4 - len(return_interm_indices):]

        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Sequential):
    """
        将backbone和position encoding集成在一个nn.Module里,使得向前过程中更方便的使用两者的功能
        Joiner是nn.Sequential的子类，通过初始化，使得self[0]是backbone，self[1]是position encoding。
        前向过程就是对backbone的每层输出都进行位置编码，最终返回backbone的输出及对应的位置编码结果。
    """

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        """
        tensor_list: pad预处理之后的图像信息
        tensor_list.tensors: [bs, 3, 608, 810]预处理后的图片数据 对于小图片而言多余部分用0填充
        tensor_list.mask: [bs, 608, 810] 用于记录矩阵中哪些地方是填充的（原图部分值为False，填充部分值为True）
        """
        # backbone的输出
        # 原图经过backbone前向传播
        # xs: '0' = NestedTensor: tensors[bs, 2048, H, W] + mask[bs, H, W]
        xs = self[0](tensor_list)  # self[0]是resnet
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():  # 逐层添加
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))  # self[1]是position encoding
        # out: list{0: tensor=[bs,2048,H,W] + mask=[bs,H,W]}  经过backbone resnet50 block5输出的结果
        # pos: list{0: [bs,256,H,W]}  位置编码
        return out, pos


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone: 
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords: 
        - use_checkpoint: for swin only for now
    """
    # 对backbone输出的特征图进行位置编码,用于后续Transformer部分
    position_embedding = build_position_encoding(args)
    # 是否需要训练backbone(即是否采用预训练backbone)
    train_backbone = args.lr_backbone > 0
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]
    backbone_freeze_keywords = args.backbone_freeze_keywords
    use_checkpoint = getattr(args, 'use_checkpoint', False)

    if args.backbone in ['resnet50', 'resnet101']:
        backbone = Backbone(args.backbone, train_backbone, args.dilation,
                            return_interm_indices,
                            batch_norm=FrozenBatchNorm2d)
        bb_num_channels = backbone.num_channels
    elif args.backbone in ['swin_T_224_1k', 'swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k']:
        pretrain_img_size = int(args.backbone.split('_')[-2])
        backbone = build_swin_transformer(args.backbone,
                                          pretrain_img_size=pretrain_img_size,
                                          out_indices=tuple(return_interm_indices),
                                          dilation=args.dilation, use_checkpoint=use_checkpoint)

        # freeze some layers
        if backbone_freeze_keywords is not None:
            for name, parameter in backbone.named_parameters():
                for keyword in backbone_freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        break

        pretrained_dir = args.backbone_dir
        PTDICT = {
            'swin_T_224_1k': 'swin_tiny_patch4_window7_224.pth',
            'swin_B_384_22k': 'swin_base_patch4_window12_384.pth',
            'swin_L_384_22k': 'swin_large_patch4_window12_384_22k.pth',
        }
        pretrainedpath = os.path.join(pretrained_dir, PTDICT[args.backbone])
        checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
        from collections import OrderedDict

        def key_select_function(keyname):
            if 'head' in keyname:
                return False
            if args.dilation and 'layers.3' in keyname:
                return False
            return True

        _tmp_st = OrderedDict({k: v for k, v in clean_state_dict(checkpoint).items() if key_select_function(k)})
        _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
        print(str(_tmp_st_output))
        bb_num_channels = backbone.num_features[4 - len(return_interm_indices):]
    elif args.backbone in ['convnext_xlarge_22k']:
        backbone = build_convnext(modelname=args.backbone, pretrained=True, out_indices=tuple(return_interm_indices),
                                  backbone_dir=args.backbone_dir)
        bb_num_channels = backbone.dims[4 - len(return_interm_indices):]
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(
        return_interm_indices), \
        f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    # 将backbone和位置编码集合在一个model
    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(bb_num_channels, List), "bb_num_channels is expected to be a List but {}".format(
        type(bb_num_channels))
    return model
