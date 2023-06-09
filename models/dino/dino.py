# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util.util_yolo import reshape_tensor_32
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import xyxy2xywhn, xywhn2xyxy
from yolov5.utils.metrics import box_iou
from .backbone import build_backbone
from .deformable_transformer import build_deformable_transformer
from .dn_components import prepare_for_cdn, dn_post_process
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .utils import sigmoid_focal_loss, MLP
from ..registry import MODULE_BUILD_FUNCS


# noinspection PyUnboundLocalVariable
class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """

    def __init__(self,
                 backbone,
                 transformer,
                 hidden_dim,
                 num_classes,
                 num_queries,
                 aux_loss=False,
                 iter_update=False,
                 query_dim=2,
                 random_refpoints_xy=False,
                 fix_refpoints_hw=-1,
                 num_feature_levels=1,
                 nheads=8,
                 two_stage_type='no',  # ['no', 'standard']
                 decoder_sa_type='sa',
                 num_patterns=0,
                 dn_number=100,
                 dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5,
                 dn_labelbook_size=100,
                 ):
        r""" Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()

        self.backbone = backbone  # backbone Joiner  0 Backbone + 1 PositionEmbeddingSine
        self.transformer = transformer
        self.model_yolo = DetectMultiBackend("C:\BaiduSyncdisk\WorkSpace\Python\yolov5\ip102_best.pt")

        # TODO 抽离成为Neck层 抽离后预训练模型将无法加载
        # 准备投影层, 将通道数高于256的特征投影至256通道. backbone->降维->transformer
        # 3个1x1conv + 1个3x3conv
        num_backbone_outs = len(backbone.num_channels)
        input_proj_list = []
        for _ in range(num_backbone_outs):  # 3个1x1conv
            in_channels = backbone.num_channels[_]  # 512  1024  2048
            input_proj_list.append(nn.Sequential(  # conv1x1  -> 256 channel
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            ))
        for _ in range(num_feature_levels - num_backbone_outs):  # 1个3x3conv
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),  # 3x3conv s=2 -> 256channel
                nn.GroupNorm(32, hidden_dim),
            ))
            in_channels = hidden_dim
        # 用于降维
        self.input_proj = nn.ModuleList(input_proj_list)

        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim  # 256
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        self.aux_loss = aux_loss  # True 计算辅助损失  6个decoder总损失

        self.iter_update = iter_update

        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        # 初始化
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_bbox_embed for _ in range(transformer.num_decoder_layers)]
        class_embed_layerlist = [_class_embed for _ in range(transformer.num_decoder_layers)]
        # 回归
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        # 分类
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type

        self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
        self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
        self.refpoint_embed = None
        self.decoder_sa_type = decoder_sa_type
        for layer in self.transformer.decoder.layers:
            layer.label_embedding = None
        self.label_embedding = None
        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def forward(self, samples: NestedTensor, targets: List = None):
        r"""
            参数samples必须是NestedTensor类型的, 它包含:
                - samples.tensor: 一个批量的图像 (batch_size, 3, H, W)
                - samples.mask: 一个由0,1构成的tensor (batch_size, H, W) 当值为1是表明此像素是通过padding得到的无意义像素
            参数targets必须是一个List元素为Tensor:
                - boxes: (x, y, h, w) 归一化尺寸
                - labels: label_id
                - image_id: image_id
                - area: 
                - iscrowd: 
                - origsize: (h, w) 原始尺寸
                - size: 

            翻译一个包含一下元素的字典:
                - pred_logits: 所有query的类别预测结果(包含空类别)
                - pred_boxes: 所有query的框坐标预测结果(X, Y, W, H) 这些值是被归一化的在[0, 1]之间
                - aux_outputs: 可选,仅在辅助损耗激活时返回。这是一个列表. 字典包含每个解码器层的上述两个密钥
        """
        # 判断samples是否是NestedTensor类型,如果不是就转成NestedTensor类型
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # features 一个由图片经过Backbone不同层抽取后的特征组成的长为3的List,内有元素有NestedTensor构成
        # [((batch_size, 512, H/8, W/8), (batch_size, H/8, W/8)),
        # ((batch_size, 1024, H/16, W/16), (batch_size, H/16, W/16)),
        # ((batch_size, 2048, H/32, W/32), (batch_size, H/32, W/32))]
        # poss 位置编码 position encoding num_pos_feats默认为256
        # [(batch_size, 256, H/8, W/8),
        # (batch_size, 256, H/16, W/16),
        # (batch_size, 256, H/32, W/32)]
        features, poss = self.backbone(samples)

        # YOLO相关
        self.model_yolo.eval()
        with torch.no_grad():
            t, h, w = reshape_tensor_32(samples.tensors)
            pred = self.model_yolo(t)
            bs = samples.tensors.shape[0]
            yolo_ref_points = torch.zeros((bs, 100, 4))
            try:
                if self.training:
                    for i in range(0, bs):
                        t_pred = pred[0][i, :, :4]
                        target = xywhn2xyxy(targets[i]["boxes"], w, h)
                        iou = box_iou(t_pred, target)
                        topk_value, topk_indices = iou.topk(100, 0)
                        topk_pred = t_pred.index_select(0, topk_indices[:, 0])  # 只考虑第一个检测框,无法做多目标
                        yolo_ref_points[i, :, :] = topk_pred[:][:]
                        yolo_ref_points = xyxy2xywhn(yolo_ref_points, w, h)
                else:
                    for i in range(0, bs):
                        conf = pred[0][i, :, 4]
                        topk_value, topk_indices = conf.topk(100, 0)
                        topk_pred = pred[0][i, :, :4].index_select(0, topk_indices[:, 0])
                        yolo_ref_points[i, :, :] = topk_pred[:][:]
                        yolo_ref_points = xyxy2xywhn(yolo_ref_points, w, h)
            except Exception as e:
                print(e)
                yolo_ref_points = None
            # print(pred[0].shape)
            # pred = non_max_suppression(pred, 0.01, 0.1, None, False, max_det=100)
            # print(pred)
            # yolo_ref_points = torch.stack(pred, dim=0)[..., :4]
            # print(yolo_ref_points)
            # yolo_ref_points = torch.clamp(xyxy2xywhn(yolo_ref_points, h, w), 0, 1)
            # print(yolo_ref_points)
            # yolo_ref_points = inverse_sigmoid(yolo_ref_points)
            # print(yolo_ref_points)
            # print(targets[0]["boxes"])

        srcs = []
        masks = []
        # TODO Neck层
        # 多尺度特征图降维,通道数降至256
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # 如果尺度类别不够,再通过降维获得新的尺度H,W再减半[1,2048,H/64,W/64]
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                # 创建对应的遮罩
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # 创建对应的位置编码 backbone位置编码 decoder使用
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        # 生成Transformer参数
        if self.dn_number > 0 or targets is not None:  # 训练
            input_query_label, input_query_bbox, attn_mask, dn_meta = \
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training,
                                num_queries=self.num_queries,
                                num_classes=self.num_classes,
                                hidden_dim=self.hidden_dim,
                                label_enc=self.label_enc)
        else:  # 推理
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(multi_level_feats=srcs,
                                                                             multi_level_masks=masks,
                                                                             multi_level_pos_embeds=poss,
                                                                             refpoint_embed=input_query_bbox,
                                                                             tgt=input_query_label,
                                                                             attn_mask=attn_mask,
                                                                             yolo_ref_points=yolo_ref_points)
        # In case num object=0
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        # 回归结果 边框
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)  # 边框
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # 回归结果 类别
        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        if self.dn_number > 0 and dn_meta is not None:  # 训练进入,推理dn_meta=None
            outputs_class, outputs_coord_list = \
                dn_post_process(outputs_class, outputs_coord_list,
                                dn_meta, self.aux_loss, self._set_aux_loss)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # for encoder output
        # prepare intermediate outputs
        interm_coord = ref_enc[-1]  # 回归结果 边框
        interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])  # 回归结果 类别
        out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
        out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

        # prepare enc outputs
        if hs_enc.shape[0] > 1:  # 默认不进入
            enc_outputs_coord = []
            enc_outputs_class = []
            for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(
                    zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                enc_outputs_coord.append(layer_enc_outputs_coord)
                enc_outputs_class.append(layer_enc_outputs_class)

            out['enc_outputs'] = [
                {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
            ]

        out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    # [num_all_gt]  记录每个预测都是来自哪张图片的 idx
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    # 记录匹配到的预测框的idx
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    # [num_all_gt]  记录每个gt都是来自哪张图片的 idx
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


def loss_boxes(outputs, targets, indices, num_boxes):
    """
    Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
    targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
    The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    targets：'boxes'=[3,4] labels=[3] ...
    indices： [3] 如：5,35,63  匹配好的3个预测框idx
    num_boxes：当前batch的所有gt个数
    """
    assert 'pred_boxes' in outputs
    # idx tuple:2  0=[num_all_gt] 记录每个gt属于哪张图片  1=[num_all_gt] 记录每个匹配到的预测框的index
    idx = _get_src_permutation_idx(indices)
    # [all_gt_num, 4]  这个batch的所有正样本的预测框坐标
    src_boxes = outputs['pred_boxes'][idx]
    # [all_gt_num, 4]  这个batch的所有gt框坐标
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    # 计算L1损失
    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

    losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

    # 计算GIOU损失
    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(target_boxes)))
    losses['loss_giou'] = loss_giou.sum() / num_boxes

    # calculate the x,y and h,w loss
    with torch.no_grad():
        losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

    # 'loss_bbox': L1回归损失   'loss_giou': giou回归损失
    return losses


def loss_masks(outputs, targets, indices, num_boxes):
    """Compute the losses related to the masks: the focal loss and the dice loss.
       targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    """
    assert "pred_masks" in outputs

    src_idx = _get_src_permutation_idx(indices)
    tgt_idx = _get_tgt_permutation_idx(indices)
    src_masks = outputs["pred_masks"]
    src_masks = src_masks[src_idx]
    masks = [t["masks"] for t in targets]
    target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
    target_masks = target_masks.to(src_masks)
    target_masks = target_masks[tgt_idx]

    # upsample predictions to the target size
    src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                            mode="bilinear", align_corners=False)
    src_masks = src_masks[:, 0].flatten(1)

    target_masks = target_masks.flatten(1)
    target_masks = target_masks.view(src_masks.shape)
    losses = {
        "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
        "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
    }
    return losses


# noinspection PyUnresolvedReferences
class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes  # 数据集类别数
        self.matcher = matcher  # HungarianMatcher()  匈牙利算法 二分图匹配
        self.weight_dict = weight_dict  # dict: 18  3x6  6个decoder的损失权重   6*(loss_ce+loss_giou+loss_bbox)
        self.losses = losses  # list: 3  ['labels', 'boxes', 'cardinality']
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        """
        Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        outputs：'pred_logits'=[bs, 100, 92] 'pred_boxes'=[bs, 100, 4] 'aux_outputs'=5*([bs, 100, 92]+[bs, 100, 4])
        targets：'boxes'=[3,4] labels=[3] ...
        indices： [3] 如：5,35,63  匹配好的3个预测框idx
        num_boxes：当前batch的所有gt个数
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # 分类：[bs, 100, 92类别]

        # idx tuple:2  0=[num_all_gt] 记录每个gt属于哪张图片  1=[num_all_gt] 记录每个匹配到的预测框的index
        idx = _get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 正样本+负样本  上面匹配到的预测框作为正样本 正常的idx  而100个中没有匹配到的预测框作为负样本(idx=91 背景类)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        # 分类损失 = 正样本 + 负样本
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        # 日志 记录分类误差
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        # losses: 'loss_ce': 分类损失
        #         'class_error':Top-1精度 即预测概率最大的那个类别与对应被分配的GT类别是否一致  这部分仅用于日志显示 并不参与模型训练
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': loss_boxes,
            'masks': loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # noinspection PyUnboundLocalVariable
    def forward(self, outputs, targets, return_indices=False):
        """
        This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
             outputs: dict of tensors, see the output specification of the model for the format
                      dict: 'pred_logits'=Tensor[bs, 100, 92个class]  'pred_boxes'=Tensor[bs, 100, 4]  最后一个decoder层输出
                             'aux_output'={list:5}  0-4  每个都是dict:2 pred_logits+pred_boxes 表示5个decoder前面层的输出
             targets: list of dicts, such that len(targets) == batch_size.   list: bs
                      每张图片包含以下信息：'boxes'、'labels'、'image_id'、'area'、'iscrowd'、'orig_size'、'size'
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # dict: 5   最后一个decoder层输出  pred_logits, pred_boxes, interm_outputs, interm_outputs_for_matching_pre, dn_meta
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device = next(iter(outputs.values())).device
        # 匈牙利算法  解决二分图匹配问题  从100个预测框中找到和N个gt框一一对应的预测框  其他的100-N个都变为背景
        # Retrieve the matching between the outputs of the last layer and the targets  list:1
        # tuple: 2    0=Tensor3=Tensor[5, 35, 63]  匹配到的3个预测框  其他的97个预测框都是背景
        #             1=Tensor3=Tensor[1, 0, 2]    对应的三个gt框
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)  # int 统计这整个batch的所有图片的gt总个数  3
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        # 计算最后层decoder损失  Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'labels' in loss:
                    kwargs = {'log': False}
                l_dict.update(
                    self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes * scalar, **kwargs))

            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 计算前面5层decoder损失  累加到一起  得到最终的losses
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)  # 同样匈牙利算法匹配
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:  # 计算各个loss
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                    l_dict = {}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes * scalar,
                                                    **kwargs))

                    l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        # 参加权重更新的损失：losses: 'loss_ce' + 'loss_bbox' + 'loss_giou'    用于log日志: 'class_error' + 'cardinality_error'
        return losses

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """
        Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            outputs: raw outputs of the model
                     0 pred_logits 分类头输出[bs, 100, 92(类别数)]
                     1 pred_boxes 回归头输出[bs, 100, 4]
                     2 aux_outputs list: 5  前5个decoder层输出 5个pred_logits[bs, 100, 92(类别数)] 和 5个pred_boxes[bs, 100, 4]
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        # out_logits：[bs, 100, 92(类别数)]
        # out_bbox：[bs, 100, 4]
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # [bs, 100, 92]  对每个预测框的类别概率取softmax
        prob = out_logits.sigmoid()
        # prob[..., :-1]: [bs, 100, 92] -> [bs, 100, 91]  删除背景
        # .max(-1): scores=[bs, 100]  100个预测框属于最大概率类别的概率
        #           labels=[bs, 100]  100个预测框的类别
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates  bs张图片的宽和高
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]  # 归一化坐标 -> 绝对位置坐标(相对于原图的坐标)  [bs, 100, 4]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b, s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in
                       zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        # list: bs    每个list都是一个dict  包括'scores'  'labels'  'boxes'三个字段
        # scores = Tensor[100,]  这张图片预测的100个预测框概率分数
        # labels = Tensor[100,]  这张图片预测的100个预测框所属类别idx
        # boxes = Tensor[100, 4] 这张图片预测的100个预测框的绝对位置坐标(相对这张图片的原图大小的坐标)
        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name='dino')
def build_dino(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    # if args.dataset_file == 'o365':
    #     num_classes = 366
    # if args.dataset_file == 'vanke':
    #     num_classes = 51

    # 最大类别ID+1
    num_classes = args.num_classes
    # 设置计算位置
    device = torch.device(args.device)
    # 搭建backbone resnet + PositionEmbeddingSine
    backbone = build_backbone(args)
    # 搭建transformer
    transformer = build_deformable_transformer(args)

    try:
        dn_labelbook_size = args.dn_labelbook_size
        if dn_labelbook_size < num_classes:
            dn_labelbook_size = num_classes
    except:
        dn_labelbook_size = num_classes

    # 搭建整个DINO模型
    model = DINO(
        backbone,
        transformer,
        args.hidden_dim,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        two_stage_type=args.two_stage_type,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
    )
    # 是否需要额外的分割任务
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    # HungarianMatcher()  二分图匹配
    matcher = build_matcher(args)

    # prepare weight dict
    # 损失权重
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_giou': args.giou_loss_coef}
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    if args.aux_loss:  # 辅助损失  每个decoder都参与计算损失  True
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update(
            {k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]

    # 定义损失函数
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             )
    criterion.to(device)

    # 定义后处理
    postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
