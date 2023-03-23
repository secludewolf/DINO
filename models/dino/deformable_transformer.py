# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import Optional

import torch
from torch import nn, Tensor

from util.misc import inverse_sigmoid
from .utils import gen_encoder_output_proposals, MLP, _get_activation_fn, gen_sineembed_for_position
from .ops.modules import MSDeformAttn


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_queries=300,
                 num_encoder_layers=6,
                 num_unicoder_layers=0,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.0,
                 activation="relu",
                 query_dim=4,
                 num_patterns=0,
                 num_feature_levels=1,
                 enc_n_points=4,
                 dec_n_points=4,
                 random_refpoints_xy=False,
                 module_seq=None):
        super().__init__()
        """
        d_model: 编码器里面mlp（前馈神经网络  2个linear层）的hidden dim 512
        nhead: 多头注意力头数 8
        num_encoder_layers: encoder的层数 6
        num_decoder_layers: decoder的层数 6
        dim_feedforward: 前馈神经网络的维度 2048
        dropout: 0.1
        activation: 激活函数类型 relu
        return_intermediate_dec: 是否返回decoder中间层结果  False
        """
        # 初始化一个encoderLayer
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead, enc_n_points
        )
        # 创建整个Encoder层  6个encoder层堆叠
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries
        )
        # 初始化一个decoderLayer
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead, dec_n_points,
            module_seq
        )
        # 创建整个Decoder层  6个decoder层堆叠
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            nn.LayerNorm(d_model),
            d_model,
            query_dim,
            num_feature_levels
        )

        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.d_model = d_model  # 编码器里面mlp的hidden dim 512
        self.nhead = nhead  # 多头注意力头数 8
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        self.num_patterns = num_patterns
        # scale-level position embedding  [4, 256]
        # 因为deformable detr用到了多尺度特征  经过backbone会生成4个不同尺度的特征图  但是如果还是使用原先的sine position embedding
        # detr是针对h和w进行编码的 不同位置的特征点会对应不同的编码值 但是deformable detr不同的特征图的不同位置就有可能会产生相同的位置编码，就无法区分了
        # 为了解决这个问题，这里引入level_embed这个遍历  不同层的特征图会有不同的level_embed 再让原先的每层位置编码+每层的level_embed
        # 这样就很好的区分不同层的位置编码了  而且这个level_embed是可学习的
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.tgt_embed = nn.Embedding(self.num_queries, d_model)
        nn.init.normal_(self.tgt_embed.weight.data)

        # anchor selection at the output of encoder
        # 对Encoder输出memory进行处理：全连接层 + 层归一化
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)

        # evolution of anchors
        self._reset_parameters()
        # 外部定义 init_ref_points
        self.refpoint_embed = None
        # 外部定义 DINO
        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def init_ref_points(self, use_num_queries):
        # 参考anchor detr, 类似于Anchor的作用
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)  # 在0-1之间随机生成
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])  # 反归一化
            self.refpoint_embed.weight.data[:, :2].requires_grad = False  # 不进行梯度计算

    def forward(self, multi_level_feats, multi_level_masks, multi_level_pos_embeds, refpoint_embed, tgt, attn_mask=None,
                yolo_ref_points=None):
        """
        Input:num_dn/ d_model =? hidden_dim
            - srcs: List of multi features [batch_size, channel, height, width]
            - masks: List of multi masks [batch_size, height, width]
            - refpoint_embed: [batch_size, num_queries, 4]. None in infer
            - pos_embeds: List of multi pos embeds [batch_size, channel, height, width]
            - tgt: [batch_size, num_queries, d_model]. None in infer
        """
        """
        Input:num_dn/ d_model =? hidden_dim
            - srcs: 一个由不同尺度特征图Tensor构成的List [batch_size, channel, height, width]
            - masks: 一个由不同尺度特征图对应遮罩的List [batch_size, height, width]
            - refpoint_embed: 新增去噪任务与推理任务的归一化坐标 [batch_size, num_queries, 4]. 仅在训练中存在
            - pos_embeds: List of multi pos embeds [batch_size, channel, height, width]
            - tgt: 新增去噪任务与推理任务的标签Embedding [batch_size, num_queries, d_model]. 仅在训练中存在
            - attn_mask: 新增去噪任务与推理任务的遮罩, 防止偷看其他数据 [query_num + dn_number * 2, query_num + dn_number * 2] 仅在训练任务中存在
        """
        # 准备encoder的输入参数
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        bs, c, h, w = (0, 0, 0, 0)
        for lvl, (src, mask, pos_embed) in enumerate(zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)):
            bs, c, h, w = src.shape  # batch_size channel h w
            spatial_shape = (h, w)  # 特征图shape
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # [bs,c,h,w] -> [bs,h*w,c]
            mask = mask.flatten(1)  # [bs,h,w] -> [bs, h*w]
            # pos_embed: detr的位置编码 仅仅可以区分h,w的位置 因此对应不同的特征图有相同的h、w位置的话，是无法区分的
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [bs,c,h,w] -> [bs,h*w,c]
            # scale-level position embedding  [bs,hxw,c] + [1,1,c] -> [bs,hxw,c]
            # 每一层所有位置加上相同的level_embed 且 不同层的level_embed不同
            # 所以这里pos_embed + level_embed，这样即使不同层特征有相同的w和h，那么也会产生不同的lvl_pos_embed  这样就可以区分了
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # list4[bs, H/8 * W/8, 256] [bs, H/16 * W/16, 256] [bs, H/32 * W/32, 256] [bs, H/64 * W/64, 256] -> [bs, K, 256]
        # K =  H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{h*w}, c
        # list4[bs, H/8 * W/8] [bs, H/16 * W/16] [bs, H/32 * W/32] [bs, H/64 * W/64] -> [bs, K]
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        # list4[bs, H/8 * W/8, 256] [bs, H/16 * W/16, 256] [bs, H/32 * W/32, 256] [bs, H/64 * W/64, 256] -> [bs, K, 256]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        # [4, h+w]  4个特征图的高和宽
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # 不同尺度特征图对应被flatten的那个维度的起始索引  Tensor[4]  如[0,15100,18900,19850]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 各尺度特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
        valid_ratios = torch.stack([get_valid_ratio(m) for m in multi_level_masks], 1)
        #########################################################
        # Begin Encoder
        #########################################################
        # [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]

        memory = self.encoder(
            query=src_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################

        # 为decoder的输入作准备：得到参考点、query embedding(tgt)和query pos(query_embed)
        # one-stage和two-stage的生成方式不同
        # two-stage: 参考点=Encoder预测的top-k（300个）得分最高的proposal boxes,然后对参考点进行位置嵌入生成query和query pos
        # one-stage: query和query pos就是预设的query_embed,然后将query_embed经过全连接层输出2d参考点（归一化的中心坐标）

        '''处理encoder输出结果'''
        input_hw = None
        # (bs, \sum{hw}, c)
        # 对memory进行处理得到output_memory: [bs, H/8 * W/8 + ... + H/64 * W/64, 256]
        # 并生成初步output_proposals: [bs, H/8 * W/8 + ... + H/64 * W/64, 4]  其实就是特征图上的一个个的点坐标
        output_memory, output_proposals = gen_encoder_output_proposals(
            memory,
            mask_flatten,
            spatial_shapes,
            input_hw
        )
        # 对encoder输出进行处理：全连接层 + LayerNorm
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        # hack implementation for two-stage Deformable DETR
        # 多分类：[bs, H/8 * W/8 + ... + H/64 * W/64, 256] -> [bs, H/8 * W/8 + ... + H/64 * W/64, 91]
        # 把每个特征点头还原分类  # 其实个人觉得这里直接进行一个二分类足够了
        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
        # 回归：预测偏移量 + 参考点坐标   [bs, H/8 * W/8 + ... + H/64 * W/64, 4]
        # 还原所有检测框  # two-stage 必须和 iterative bounding box refinement一起使用 不然bbox_embed=None 报错
        enc_outputs_coord_unselected = self.enc_out_bbox_embed(
            output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
        # 保留前多少个query  # 得到参考点reference_points/先验框
        topk = self.num_queries
        # 直接用第一个类别的预测结果来算top-k，代表二分类
        # 如果不使用iterative bounding box refinement那么所有class_embed共享参数 导致第二阶段对解码输出进行分类时都会偏向于第一个类别
        # 获取前900个检测结果的位置  # topk_proposals: [bs, 900]  top900 index
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]  # bs, nq

        # gather boxes
        # 根据前九百个位置获取对应的参考点  # topk_coords_unact: top300个分类得分最高的index对应的预测bbox [bs, 900, 4]
        refpoint_embed_undetach = torch.gather(  # unsigmoid
            enc_outputs_coord_unselected,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )
        # 以先验框的形式存在  取消梯度
        refpoint_embed_ = refpoint_embed_undetach.detach()
        # 得到归一化参考点坐标  最终会送到decoder中作为初始的参考点
        init_box_proposal = torch.gather(
            output_proposals,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        ).sigmoid()

        # gather tgt  # 根据前九百个位置获取query
        tgt_undetach = torch.gather(
            output_memory,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
        )
        tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)  # nq, bs, d_model

        if refpoint_embed is not None:  # 训练 query + cdn_query
            refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_],
                                       dim=1)  # 将正常query与cdn正负情绪query的bbox拼接900+200
            tgt = torch.cat([tgt, tgt_], dim=1)  # 将初始化query与cdn正负情绪query拼接900+200
        else:  # 推理 query
            refpoint_embed, tgt = refpoint_embed_, tgt_

        # 设置yolo参考点
        if yolo_ref_points is not None:
            bs = yolo_ref_points.shape[0]
            yolo_length = yolo_ref_points.shape[1]
            resource_length = refpoint_embed.shape[1]
            for ibs in range(0, bs):
                for i in range(0, yolo_length):
                    # print(i, yolo_ref_points[ibs][i][:])
                    refpoint_embed[ibs][resource_length - i - 1] = yolo_ref_points[ibs][i][:]

        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model 1, 1100, 256
        # - refpoint_embed(unsigmoid): bs, NQ, d_model 1, 1100, 4
        ######################################################### 

        #########################################################
        # Begin Decoder
        #########################################################
        # tgt: 初始化query embedding [bs, 900 + 200, 256]
        # memory: Encoder输出结果 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        # mask_flatten: 4个特征层flatten后的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        # lvl_pos_embed_flatten: query pos[bs, 300, 256]
        # reference_points: 由query pos接一个全连接层 再归一化后的参考点中心坐标 [bs, 900 + 200, 2]  two-stage=[bs, 900 + 200, 4]
        # level_start_index: [4, ] 4个特征层flatten后的开始index
        # spatial_shapes: [4, 2] 4个特征层的shape
        # valid_ratios: [bs, 4, 2] padding比例 全1
        # attn_mask: [900 + 200, 900 + 200] query遮罩,防止偷看
        # hs: 6层decoder输出 [n_decoder, bs, num_query, d_model] = [6, bs, 900 + 200, 256]
        # inter_references: 6层decoder学习到的参考点归一化中心坐标  [6, bs, 900 + 200, 2]
        #                   one-stage=[n_decoder, bs, num_query, 2]  two-stage=[n_decoder, bs, num_query, 4]
        hs, references = self.decoder(
            query=tgt.transpose(0, 1),
            key=memory.transpose(0, 1),
            value=memory.transpose(0, 1),
            query_pos=None,
            key_padding_mask=mask_flatten,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        hs_enc = tgt_undetach.unsqueeze(0)
        ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################        

        # hs: 6层decoder输出 [n_decoder, bs, num_query, d_model] = [6, bs, 300, 256]
        # init_reference_out: 初始化的参考点归一化中心坐标 [bs, 300, 2]
        # inter_references: 6层decoder学习到的参考点归一化中心坐标  [6, bs, 300, 2]
        #                   one-stage=[n_decoder, bs, num_query, 2]  two-stage=[n_decoder, bs, num_query, 4]
        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates.
        # (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


class TransformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer,
                 num_layers,
                 d_model=256,
                 num_queries=300,
                 ):
        super().__init__()
        # prepare layers
        if num_layers > 0:
            # 6层DeformableTransformerEncoderLayer
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=False)
        else:
            self.layers = []
            del encoder_layer

        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        生成参考点   reference points  为什么参考点是中心点？  为什么要归一化？
        spatial_shapes: 多尺度feature map对应的h,w，shape为[num_level,2]
        valid_ratios: 多尺度feature map对应的mask中有效的宽高比，shape为[bs, num_levels, 2]  如全是1
        device: cuda:0
        """
        reference_points_list = []
        # 遍历4个特征图的shape  比如 H_=100  W_=150
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 0.5 -> 99.5 取100个点  0.5 1.5 2.5 ... 99.5
            # 0.5 -> 149.5 取150个点 0.5 1.5 2.5 ... 149.5
            # ref_y: [100, 150]  第一行：150个0.5  第二行：150个1.5 ... 第100行：150个99.5
            # ref_x: [100, 150]  第一行：0.5 1.5...149.5   100行全部相同
            # 对于每一层feature map初始化每个参考点中心横纵坐标，加减0.5是确保每个初始点是在每个pixel的中心，例如[0.5,1.5,2.5, ...]
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # 将横纵坐标进行归一化，处理成0-1之间的数 [h, w] -> [bs, hw]  150个0.5 + 150个1.5 + ... + 150个99.5 -> 除以100 归一化
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            # [h, w] -> [bs, hw]  100个: 0.5 1.5 ... 149.5  -> 除以150 归一化
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # 得到每一层feature map对应的reference point，即ref，shape为[B, flatten_W*flatten_H, 2] 每一项都是xy
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # list4: [bs, H/8*W/8, 2] + [bs, H/16*W/16, 2] + [bs, H/32*W/32, 2] + [bs, H/64*W/64, 2] ->
        # 将所有尺度的feature map对应的reference point在第一维合并，得到[bs, N, 2] [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2]
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points: [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2] -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 1, 2]
        # valid_ratios: [1, 4, 2] -> [1, 1, 4, 2]
        # 从[bs, N, 2]扩充尺度到[bs, N, num_level, 2] 复制4份 每个特征点都有4个归一化参考点 -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 4, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # 4个flatten后特征图的归一化参考点坐标
        return reference_points

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                query_pos: Tensor,
                query_key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]

            - ref_token_coord: bs, nq, 4
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus: 
            - output: [bs, sum(hi*wi), 256]
        """
        """
        src: 多尺度特征图(4个flatten后的特征图)  [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        spatial_shapes: 4个特征图的shape [4, 2]
        level_start_index: [4] 4个flatten后特征图对应被flatten后的起始索引  如[0,15100,18900,19850]
        valid_ratios: 4个特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
        pos: 4个flatten后特征图对应的位置编码（多尺度位置编码） [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        padding_mask: 4个flatten后特征图的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        # preparation and reshape
        # 4个flatten后特征图的归一化参考点坐标 每个特征点有4个参考点 xy坐标 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 4, 2]
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=query.device)
        for layer_id, layer in enumerate(self.layers):
            query = layer(query=query,
                          key=key,
                          valule=value,
                          query_pos=query_pos,
                          reference_points=reference_points,
                          spatial_shapes=spatial_shapes,
                          level_start_index=level_start_index,
                          key_padding_mask=query_key_padding_mask,
                          )
        # 经过6层encoder增强后的新特征  每一层不断学习特征层中每个位置和4个采样点的相关性，最终输出的特征是增强后的特征图
        # [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        return query


class TransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 d_model=256,
                 query_dim=4,
                 num_feature_levels=1,
                 ):
        super().__init__()
        # 6层DeformableTransformerDecoderLayer
        self.layers = _get_clones(decoder_layer, num_layers, layer_share=False)
        self.num_layers = num_layers  # 6
        self.norm = norm
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None
        self.query_scale = None
        self.bbox_embed = None  # 策略1  iterative bounding box refinement
        self.class_embed = None  # 策略2  two-stage Deformable DETR
        self.d_model = d_model
        self.ref_anchor_head = None

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_padding_mask: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                level_start_index: Optional[Tensor] = None,  # num_levels
                valid_ratios: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
            tgt: [900 + 200, bs, 256] 预设的query embedding
            memory: [H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, bs, 256] 来自backbone的多层特征信息
            tgt_mask: [1100, 1100] query遮罩
            memory_key_padding_mask: [batch, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, bs, 256] 来自backbone的遮罩信息
            pos: [H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, bs, 256] 来自backbone的位置信息
            refpoints_unsigmoid: [1100, 1, 4] query的参考点 替代了query pos
            level_start_index: [4,] 每层的起始位置
            spatial_shapes: [4, 2] 每层的实际大小
            valid_raitos: [bs, 4, 2] 每层图片padding后实际所占的比例
        """
        output = query

        intermediate = []  # 中间各层+首尾两层=6层输出的解码结果
        reference_points = refpoints_unsigmoid.sigmoid()  # 中间各层+首尾两层输出的参考点（不断矫正）
        ref_points = [reference_points]  # 起始有一个+中间各层+首尾两层=7个参考点

        for layer_id, layer in enumerate(self.layers):
            # 得到参考点坐标
            # (1100, bs, 4, 4) 拷贝四份,每层一份
            reference_points_input = reference_points[:, :, None] \
                                     * torch.cat([valid_ratios, valid_ratios], -1)[None, :]  # nq, bs, nlevel, 4
            # 生成位置编码  根据参考点生成位置编码
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])  # nq, bs, 256*2

            # conditional query
            # 原生query位置编码
            raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            # random drop some layers if needed
            output = layer(
                query=output,  # [1100, 1, 256]上一层的输出
                key=key,  # [hw, bs, 256]
                value=value,
                query_pos=query_pos,  # 位置编码
                key_padding_mask=key_padding_mask,  # [bs, hw]
                key_level_start_index=level_start_index,  # [4,] 层起始位置
                key_spatial_shapes=spatial_shapes,  # [4, 2] 层大小
                attn_mask=attn_mask,  # [1100, 1100] 遮罩
                reference_points=reference_points_input,  # [1100, 1, 4, 4] nq, bs, nlevel, 4 参考点
            )

            # iter update
            # hack implementation for iterative bounding box refinement
            # 使用iterative bounding box refinement 这里的self.bbox_embed就不是None
            # 如果没有iterative bounding box refinement那么reference_points是不变的
            # 每层参考点都会根据上一层的输出结果进行矫正
            reference_before_sigmoid = inverse_sigmoid(reference_points)  # 还原参考点
            delta_unsig = self.bbox_embed[layer_id](output)
            outputs_unsig = delta_unsig + reference_before_sigmoid
            new_reference_points = outputs_unsig.sigmoid()
            reference_points = new_reference_points.detach()
            ref_points.append(new_reference_points)

            # 默认返回6个decoder层输出一起计算损失
            intermediate.append(self.norm(output))

        return [[itm_out.transpose(0, 1) for itm_out in intermediate],
                [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]]


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 d_ffn=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_heads=8,
                 n_points=4,
                 ):
        super().__init__()
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                query,
                key,
                value,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                key_padding_mask=None
                ):
        """
        src: 多尺度特征图(4个flatten后的特征图)  [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        reference_points: 4个flatten后特征图对应的归一化参考点坐标 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 4, 2]
        pos: 4个flatten后特征图对应的位置编码（多尺度位置编码） [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        spatial_shapes: 4个特征图的shape [4, 2]
        level_start_index: [4] 4个flatten后特征图对应被flatten后的起始索引  如[0,15100,18900,19850]
        padding_mask: 4个flatten后特征图的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        # self attention + add + norm
        # query = flatten后的多尺度特征图 + scale-level pos
        # key = 采样点  每个特征点对应周围的4个可学习的采样点
        # value = flatten后的多尺度特征图
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(query, query_pos),
            reference_points,
            query,
            spatial_shapes,
            level_start_index,
            key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn   feed forward + add + norm
        query = self.forward_ffn(query)

        # [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        return query


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 module_seq=None,
                 ):
        super().__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self,
                   query: Optional[Tensor],  # nq, bs, d_model
                   query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                   attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                   ):
        # self attention
        q = k = self.with_pos_embed(query, query_pos)
        # 第一个attention的目的：学习各个物体之间的关系/位置   可以知道图像当中哪些位置会存在物体  物体信息->tgt
        # 所以qk都是query embedding + query pos   v就是query embedding
        tgt2 = self.self_attn(q, k, query, attn_mask=attn_mask)[0]
        query = query + self.dropout2(tgt2)
        query = self.norm2(query)

        return query

    def forward_ca(self,
                   query: Optional[Tensor],  # nq, bs, d_model
                   query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                   reference_points: Optional[Tensor] = None,  # nq, bs, 4
                   key: Optional[Tensor] = None,  # hw, bs, d_model
                   key_padding_mask: Optional[Tensor] = None,
                   key_level_start_index: Optional[Tensor] = None,  # num_levels
                   key_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                   ):
        # cross attention  使用（多尺度）可变形注意力模块替代原生的Transformer交叉注意力
        # 第二个attention的目的：不断增强encoder的输出特征，将物体的信息不断加入encoder的输出特征中去，更好地表征了图像中的各个物体
        # 所以q=query embedding + query pos, k = query pos通过一个全连接层->2维, v=上一层输出的output
        tgt2 = self.cross_attn(self.with_pos_embed(query, query_pos).transpose(0, 1),
                               reference_points.transpose(0, 1).contiguous(),
                               key.transpose(0, 1), key_spatial_shapes, key_level_start_index,
                               key_padding_mask).transpose(0, 1)
        # add + norm
        query = query + self.dropout1(tgt2)
        query = self.norm1(query)
        # [bs, 300, 256]  self-attention输出特征 + cross-attention输出特征
        # 最终的特征：知道图像中物体与物体之间的位置关系 + encoder增强后的图像特征 + 图像与物体之间的关系
        return query

    def forward(self,
                query: Optional[Tensor],  # nq, bs, d_model
                key: Optional[Tensor] = None,  # hw, bs, d_model
                value: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                key_padding_mask: Optional[Tensor] = None,
                key_level_start_index: Optional[Tensor] = None,  # num_levels
                key_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                reference_points: Optional[Tensor] = None,  # nq, bs, 4
                ):
        """
        tgt: 预设的query embedding [bs, 300, 256]
        query_pos: 预设的query pos [bs, 300, 256]
        reference_points: query pos通过一个全连接层->2维  [bs, 300, 4, 2] = [bs, num_query, n_layer, 2]
                          iterative bounding box refinement时 = [bs, num_query, n_layer, 4]
        src: 第一层是encoder的输出memory 第2-6层都是上一层输出的output
        src_spatial_shapes: [4, 2] 4个特征层的原始shape
        src_level_start_index: [4,] 4个特征层flatten后的开始index
        src_padding_mask: 4个特征层flatten后的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        for funcname in self.module_seq:
            if funcname == 'ffn':
                query = self.forward_ffn(query)
            elif funcname == 'ca':
                query = self.forward_ca(query=query,
                                        query_pos=query_pos,
                                        reference_points=reference_points,
                                        key=key,
                                        key_padding_mask=key_padding_mask,
                                        key_level_start_index=key_level_start_index,
                                        key_spatial_shapes=key_spatial_shapes,
                                        )
            elif funcname == 'sa':
                query = self.forward_sa(query=query,
                                        query_pos=query_pos,
                                        attn_mask=attn_mask,
                                        )
        return query


def _get_clones(module, size, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for _ in range(size)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(size)])


def build_deformable_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        random_refpoints_xy=args.random_refpoints_xy,
        # two stage
        module_seq=args.decoder_module_seq,
    )
