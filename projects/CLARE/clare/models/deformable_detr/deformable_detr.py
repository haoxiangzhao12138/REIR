# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
import math

from ...util import box_ops
from ...util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (dice_loss, sigmoid_focal_loss, token_sigmoid_binary_focal_loss)
from torch.distributed.nn import all_gather

import copy
from fvcore.nn import giou_loss, smooth_l1_loss
import random

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

import torch
import torch.nn as nn
import torch.nn.functional as F

# class Expert(nn.Module):
#     """
#     Expert layer for Mixture-of-Experts (MoE) models.

#     Attributes:
#         w1 (nn.Module): Linear layer for input-to-hidden transformation.
#         w2 (nn.Module): Linear layer for hidden-to-output transformation.
#         w3 (nn.Module): Additional linear layer for feature transformation.
#     """
#     def __init__(self, cfg):
#         """
#         Initializes the Expert layer.

#         Args:
#             dim (int): Input and output dimensionality.
#             inter_dim (int): Hidden layer dimensionality.
#         """
#         super().__init__()
#         inter_dim = cfg.MODEL.MOE.HIDDEN_DIM
#         dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
#         output_dim = cfg.MODEL.DDETRS.HIDDEN_DIM
#         self.w1 = nn.Linear(dim, inter_dim)
#         self.w2 = nn.Linear(inter_dim, output_dim)
#         self.w3 = nn.Linear(dim, inter_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for the Expert layer.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             torch.Tensor: Output tensor after expert computation.
#         """
#         return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Expert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.inter_dim = cfg.MODEL.MOE.HIDDEN_DIM
        self.input_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        self.output_dim = cfg.MODEL.DDETRS.HIDDEN_DIM

        self.w1 = nn.Linear(self.input_dim, self.output_dim)
    
    def forward(self, x):
        return self.w1(x)




class MoETextProjection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.MODEL.MOE.NUM_EXPERTS
        self.top_k = cfg.MODEL.MOE.TOP_K  # 选择 top-k 个专家
        self.output_dim = cfg.MODEL.DDETRS.HIDDEN_DIM
        self.input_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM

        # 门控网络，决定输入应使用哪些专家
        self.gate_network = nn.Linear(self.input_dim, self.num_experts, bias=True)

        # 创建多个专家，每个专家是一个 FFN
        self.experts = nn.ModuleList([
            Expert(cfg) for _ in range(self.num_experts)
        ])

    def forward(self, x):
        """
        x: (bs, L, input_dim) 文本特征
        输出: (bs, L, output_dim) 降维后的文本特征
        """
        bs, L, _ = x.shape

        # 计算 gate 分数
        gate_logits = self.gate_network(x)  # (bs, L, num_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (bs, L, num_experts)

        # 选取 top-k 个专家
        topk_values, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # (bs, L, top_k)

        # 初始化最终输出
        output = torch.zeros(bs, L, self.output_dim, device=x.device)

        # 遍历 top-k 专家并加权求和
        for i in range(self.top_k):
            expert_idx = topk_indices[..., i]  # (bs, L)
            expert_weight = topk_values[..., i].unsqueeze(-1)  # (bs, L, 1)

            # 选择对应专家
            expert_output = torch.stack([self.experts[idx](x[b, l]) for b in range(bs) for l, idx in enumerate(expert_idx[b])])
            expert_output = expert_output.view(bs, L, self.output_dim)  # (bs, L, output_dim)

            # 加权累加
            output += expert_output * expert_weight

        return output


# To compute VL Alignment scores 
class VL_Align(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_bias = cfg.MODEL.USE_BIAS
        self.use_batch_loss = cfg.MODEL.USE_BATCH_LOSS
        self.use_siglip_loss = cfg.MODEL.USE_SIGLIP_LOSS
        self.use_contrastive_focal_loss = cfg.MODEL.USE_CONTRASTIVE_FOCAL_LOSS
        self.objects_query_num = cfg.MODEL.DDETRS.NUM_OBJECT_QUERIES
        self.dot_product_projection_image = nn.Identity()
        self.use_moe = cfg.MODEL.MOE.USE_MOE
        self.dot_product_projection_text = nn.Linear(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM, cfg.MODEL.DDETRS.HIDDEN_DIM, bias=True)

        if self.use_moe:
            self.moe_weight = cfg.MODEL.MOE.WEIGHT
            self.moe = MoETextProjection(cfg)            

        if self.use_bias:
            # initialize the bias for focal loss
            prior_prob = cfg.MODEL.DYHEAD.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.log_scale = nn.Parameter(torch.Tensor([cfg.MODEL.DYHEAD.LOG_SCALE]), requires_grad=True)
            self.bias_lang = nn.Parameter(torch.zeros(cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM), requires_grad=True) # (1024,)
            self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True) # size (1,)

        if self.use_siglip_loss:
            self.temperature = nn.Parameter(torch.ones(1) * math.log(10), requires_grad=True) 
            self.b = nn.Parameter(torch.zeros(1) * (-10), requires_grad=True)   
    
    def forward(self, x, embedding, return_feature=False):
        """
        x: visual features (bs, num_query, 256)
        embedding: language features (bs, L, 768)
        """
        # 对语言特征进行归一化
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        # 图像和文本投影
        dot_product_proj_queries = self.dot_product_projection_image(x)  # (bs, num_query, 256)
        dot_product_proj_tokens = self.dot_product_projection_text(embedding)  # (bs, L, 256)
        if self.use_moe:
            moe_output = self.moe(embedding)
            dot_product_proj_tokens = (1 - self.moe_weight) * dot_product_proj_tokens + self.moe_weight * moe_output
        
        # 如果使用 batch 内对比学习，则不使用 bias
        dot_product_logit_batch = None
        if self.use_siglip_loss and not return_feature:
            object_queries = dot_product_proj_queries[:,-self.objects_query_num:,:]  # (bs, num_query, 256)
            
            B, A, _ = object_queries.shape  # B: batch size, A: num_query
            _, L, _ = dot_product_proj_tokens.shape   # L: token length
            
            # 合并 batch
            dot_product_proj_queries_flat = object_queries.reshape(B * A, -1)   # (B*N, 256)
            dot_product_proj_tokens_flat = dot_product_proj_tokens.reshape(B * L, -1)     # (B*L, 256)

            all_text_features = torch.cat(all_gather(dot_product_proj_tokens_flat))  # (8*B, 256)
            part_image_features = dot_product_proj_queries_flat  # (B*900, 256)

            # dot_product_logit_batch = torch.matmul(part_image_features, all_text_features.transpose(0, 1))  # (B*900, 8*B)
            dot_product_logit_batch = torch.matmul(part_image_features, all_text_features.transpose(0, 1)) * self.temperature.exp() + self.b  # (B*900, 8*B)
            
            # # 无偏置情况直接计算相似度
            # dot_product_logit_batch = torch.matmul(dot_product_proj_queries_flat, dot_product_proj_tokens_flat.transpose(0, 1))  # (B*A, B*L)
            # # dot_product_logit_batch = dot_product_logit_batch.reshape(B, A, B * L)  # (B, A, B*L)
            
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                dot_product_logit_batch = torch.clamp(dot_product_logit_batch, min=-50000, max=50000)


        elif self.use_batch_loss and not return_feature:
            object_queries = dot_product_proj_queries[:,-self.objects_query_num:,:]  # (bs, num_query, 256)
            
            B, A, _ = object_queries.shape  # B: batch size, A: num_query
            _, L, _ = dot_product_proj_tokens.shape   # L: token length
            
            # 合并 batch
            dot_product_proj_queries_flat = object_queries.reshape(B * A, -1)   # (B*N, 256)
            dot_product_proj_tokens_flat = dot_product_proj_tokens.reshape(B * L, -1)     # (B*L, 256)

            dot_product_logit_batch = torch.matmul(part_image_features, all_text_features.transpose(0, 1))  # (B*900, 8*B)
            
            # 无偏置情况直接计算相似度
            dot_product_logit_batch = torch.matmul(dot_product_proj_queries_flat, dot_product_proj_tokens_flat.transpose(0, 1))  # (B*A, B*L)
            # dot_product_logit_batch = dot_product_logit_batch.reshape(B, A, B * L)  # (B, A, B*L)
            
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                dot_product_logit_batch = torch.clamp(dot_product_logit_batch, min=-50000, max=50000)   
        elif self.use_contrastive_focal_loss and not return_feature:
            object_queries = dot_product_proj_queries[:,-self.objects_query_num:,:]  # (bs, num_query, 256)
            
            B, A, _ = object_queries.shape  # B: batch size, A: num_query
            _, L, _ = dot_product_proj_tokens.shape   # L: token length
            
            # 合并 batch
            dot_product_proj_queries_flat = object_queries.reshape(B * A, -1)   # (B*N, 256)
            dot_product_proj_tokens_flat = dot_product_proj_tokens.reshape(B * L, -1)     # (B*L, 256)

            all_image_features = torch.cat(all_gather(dot_product_proj_queries_flat))  # (8*B*900, 256)
            part_text_features = dot_product_proj_tokens_flat  # (B*L, 256)
            
            dot_product_logit_batch = torch.matmul(all_image_features, part_text_features.transpose(0, 1))
                  
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                dot_product_logit_batch = torch.clamp(dot_product_logit_batch, min=-50000, max=50000)         
            
        # 原有逻辑：逐样本计算对比度
        if self.use_bias:
            dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0  # (bs, L)
            A = dot_product_proj_queries.shape[1] # num_query
            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1) # (bs, num_query, L)

            dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
            if return_feature:
                return dot_product_logit, dot_product_proj_queries, dot_product_proj_tokens, dot_product_proj_tokens_bias, self.log_scale.exp()
            return dot_product_logit, dot_product_logit_batch
        else:
            dot_product_logit = torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2))
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:
                dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
            if return_feature:
                return dot_product_logit, dot_product_proj_queries, dot_product_proj_tokens, None, None
            return dot_product_logit, dot_product_logit_batch

class Still_Classifier(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.body = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, lang_feat=None):
        return self.body(x)
        

class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, 
                 mixed_selection=False, cfg=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = VL_Align(cfg)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if cfg.MODEL.USE_IOU_BRANCH:
            self.iou_head = nn.Linear(hidden_dim, 1)
        self.num_feature_levels = num_feature_levels
        self.decouple_tgt = cfg.MODEL.DECOUPLE_TGT
        if self.decouple_tgt:
            if not two_stage:
                self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
            elif mixed_selection:
                self.query_embed = nn.Embedding(num_queries, hidden_dim)
        else:
            # language as tgt
            if not two_stage:
                self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # if cfg.MODEL.USE_IOU_BRANCH:
        #     self.iou_head.bias.data = torch.ones(1) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            if cfg.MODEL.USE_IOU_BRANCH:
                self.iou_head = _get_clones(self.iou_head, num_pred-1)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            if cfg.MODEL.USE_IOU_BRANCH:
                self.iou_head = nn.ModuleList([self.iou_head for _ in range(num_pred-1)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            if cfg.MODEL.STILL_CLS_FOR_ENCODER:
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                self.transformer.decoder.class_embed[-1] = Still_Classifier(hidden_dim) # binary classification
                self.transformer.decoder.class_embed[-1].body.bias.data = torch.ones(1) * bias_value
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.mixed_selection = mixed_selection

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class DeformableDETRDINO(nn.Module):
    # NO query_embed
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, 
                 mixed_selection=False, cfg=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = VL_Align(cfg)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if cfg.MODEL.USE_IOU_BRANCH:
            self.iou_head = nn.Linear(hidden_dim, 1)
        self.num_feature_levels = num_feature_levels
        self.decouple_tgt = cfg.MODEL.DECOUPLE_TGT

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if cfg.MODEL.USE_IOU_BRANCH:
            self.iou_head.bias.data = torch.ones(1) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            if cfg.MODEL.USE_IOU_BRANCH:
                self.iou_head = _get_clones(self.iou_head, num_pred-1)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            if cfg.MODEL.USE_IOU_BRANCH:
                self.iou_head = nn.ModuleList([self.iou_head for _ in range(num_pred-1)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            if cfg.MODEL.STILL_CLS_FOR_ENCODER:
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                self.transformer.decoder.class_embed[-1] = Still_Classifier(hidden_dim) # binary classification
                self.transformer.decoder.class_embed[-1].body.bias.data = torch.ones(1) * bias_value
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.mixed_selection = mixed_selection


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, matcher, weight_dict, losses, focal_alpha=0.25, mask_out_stride=4, ota=False, still_cls_for_encoder=False, cfg=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.mask_out_stride = mask_out_stride
        self.ota = ota
        self.still_cls_for_encoder = still_cls_for_encoder
        self.num_query = cfg.MODEL.DDETRS.NUM_OBJECT_QUERIES
        self.use_batch_loss = cfg.MODEL.USE_BATCH_LOSS
        self.use_siglip_loss = cfg.MODEL.USE_SIGLIP_LOSS
        self.use_contrastive_focal_loss = cfg.MODEL.USE_CONTRASTIVE_FOCAL_LOSS
        self.use_clip_pool_head = cfg.MODEL.USE_CLIP_POOL_HEAD
        # boxinst configs
        if cfg is not None:
            self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
            self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
            self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
            self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
            self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
            self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
            self.boxinst_topk = cfg.MODEL.BOXINST.TOPK
            self.language_backbone = cfg.MODEL.LANGUAGE_BACKBONE.NAME
            self.register_buffer("_iter", torch.zeros([1]))

    def loss_labelsVL(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        assert 'text_masks' in outputs
        src_logits = outputs['pred_logits'] # (bs, num_query, C)

        idx = self._get_src_permutation_idx(indices) # tuple (batch_idx, src_idx)
        num_boxes = len(idx[0]) if self.ota else num_boxes
        if num_boxes == 0:
            loss_ce = src_logits.sum() * 0.0
            losses = {'loss_ce': loss_ce}
            return losses
        else:
            target_classes_onehot = torch.zeros(src_logits.size(),
                                    dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # (bs, num_query, C)
            positive_map = [x["positive_map"] for x in targets] # [(N1, 256), (N2, 256)]
            # loop over batch size
            for batch_idx, (src_idxs, target_idxs) in enumerate(indices):
                # loop over objects in one image
                for (src_idx, target_idx) in zip(src_idxs, target_idxs):
                    target_classes_onehot[batch_idx, src_idx] = positive_map[batch_idx][target_idx]
            text_mask = outputs['text_masks']
            if self.language_backbone == "openclip":
                text_mask = None
            loss_ce = token_sigmoid_binary_focal_loss(src_logits, target_classes_onehot, text_mask=text_mask) / num_boxes
            losses = {'loss_ce': loss_ce}
            if log:
                raise ValueError("log is not supported.")
            return losses
    
    def loss_VL_batch(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'logits_batch' in outputs
        assert 'text_masks' in outputs
        src_logits = outputs['logits_batch'] # (bs*num_query, text_num)
        device = src_logits.device

        # 从indices获取源查询的batch_idx和src_idx
        # 假设 _get_src_permutation_idx 返回 (batch_idx, src_idx), 都是1D张量
        batch_idx, src_idx = self._get_src_permutation_idx(indices)  

        # 若使用OTA，则num_boxes为匹配的总数，否则使用传入的num_boxes
        num_boxes = len(batch_idx) if self.ota else num_boxes
        if num_boxes == 0:
            # 无匹配，返回0损失
            loss_ce = src_logits.sum() * 0.0
            losses = {'loss_batch': loss_ce}
            return losses
        
        # 将(batch_idx, src_idx)映射到扁平化后的索引 flat_src_idx
        # 原逻辑：src_logits维度 (bs * num_query, bs)
        # flat_src_idx = batch_idx * num_query + src_idx
        

        # 创建与src_logits同形状的one-hot目标张量 (bs*num_query, bs)
        target_classes_onehot = torch.zeros_like(src_logits)  # shape: (bs * num_query, bs)

        # 根据indices填充target_classes_onehot
        # indices是[(src_idxs, tgt_idxs), ...]，长度为batch_size
        # i为图片的index，(src_idxs, tgt_idxs)是该图像内的匹配对
        for i, (cur_src_idxs, cur_tgt_idxs) in enumerate(indices):
            for s_idx, t_idx in zip(cur_src_idxs, cur_tgt_idxs):
                # s_idx为image i中的query索引
                # t_idx为image i中的目标索引
                # 从targets[i]["labels"]中获取该目标对应的类标(范围[0, bs-1])
                # 将batch内的 (i, s_idx) 转换为全局索引
                global_src_idx = i * self.num_query + s_idx
                target_classes_onehot[global_src_idx, i] = 1.0

        # 计算loss，使用与之前相同的focal loss形式
        loss_ce = token_sigmoid_binary_focal_loss(
            src_logits.unsqueeze(0), target_classes_onehot.unsqueeze(0), text_mask=None
        ) / num_boxes

        losses = {'loss_batch': loss_ce}
        if log:
            raise ValueError("log is not supported.")
        return losses
    
    def sigmod_loss(self, outputs, targets, indices, num_boxes, log=False):
        assert 'logits_batch' in outputs
        assert 'text_masks' in outputs
        src_logits = outputs['logits_batch'] # (bs*num_query, gpu*bs_text_num)

        rank = torch.distributed.get_rank()
        device = src_logits.device

        # 创建与src_logits同形状的one-hot目标张量 (bs*num_query, gpu*bs_text_num)
        target_classes_onehot = -torch.ones_like(src_logits)  # shape: (bs*num_query, gpu*bs_text_num)

        # 根据indices填充target_classes_onehot
        for i, (cur_src_idxs, cur_tgt_idxs) in enumerate(indices):
            for s_idx, t_idx in zip(cur_src_idxs, cur_tgt_idxs):
                # s_idx为image i中的query索引

                # 将batch内的 (i, s_idx) 转换为全局索引
                global_src_idx = i * self.num_query + s_idx
                global_text_idx = rank * len(targets) + i
                target_classes_onehot[global_src_idx, global_text_idx] = 1.0

        # 计算loss
        loss = - torch.sum(F.logsigmoid(target_classes_onehot * src_logits)) / src_logits.size(1)
        losses = {'loss_siglip': loss}
        if log:
            raise ValueError("log is not supported.")
        return losses

    def contrastive_focal_loss(self, outputs, targets, indices, num_boxes, log=False):
        assert 'logits_batch' in outputs
        assert 'text_masks' in outputs
        src_logits = outputs['logits_batch']

        rank = torch.distributed.get_rank()
        device = src_logits.device

        # 创建与src_logits同形状的one-hot目标张量 
        target_classes_onehot = torch.zeros_like(src_logits)  # shape: 

        # 根据indices填充target_classes_onehot
        for i, (cur_src_idxs, cur_tgt_idxs) in enumerate(indices):
            for s_idx, t_idx in zip(cur_src_idxs, cur_tgt_idxs):
                # s_idx为image i中的query索引

                # 将batch内的 (i, s_idx) 转换为全局索引
                global_src_idx = rank * self.num_query * len(targets) + i * self.num_query + s_idx
                global_text_idx = i
                target_classes_onehot[global_src_idx, global_text_idx] = 1.0

        # 计算loss
        loss = token_sigmoid_binary_focal_loss(
            src_logits.unsqueeze(0), target_classes_onehot.unsqueeze(0), gamma=4.0, text_mask=None
        ) / num_boxes
        losses = {'loss_contrastive': loss}
        if log:
            raise ValueError("log is not supported.")
        return losses        
    
    
    def loss_text_image(self, outputs, targets, indices, num_boxes, log=False):
        all_image_features = torch.cat(all_gather(outputs['image_features']))
        all_text_features = torch.cat(all_gather(outputs['text_features']))

        logits = all_image_features @ all_text_features.T
        batch_size = all_image_features.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        # 计算 text-to-image 和 image-to-text 交叉熵损失
        loss_img2txt = F.cross_entropy(logits, labels)  # 图像作为查询，文本为标签
        loss_txt2img = F.cross_entropy(logits.T, labels)  # 文本作为查询，图像为标签

        losses = {'loss_text_image': (loss_img2txt + loss_txt2img) / 2}
        return losses



    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
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

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if len(target_boxes) == 0:
            losses = {}
            losses['loss_bbox'] = src_boxes.sum() * 0.0
            losses['loss_giou'] = src_boxes.sum() * 0.0
            return losses

        # box iou
        if 'pred_boxious' in outputs:
            with torch.no_grad():
                ious = compute_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes),
                                        box_ops.box_cxcywh_to_xyxy(target_boxes))                    
            tgt_iou_scores = ious
            src_iou_scores = outputs['pred_boxious'] # [B, N, 1]
            src_iou_scores = src_iou_scores[idx]
            src_iou_scores = src_iou_scores.flatten(0)
            tgt_iou_scores = tgt_iou_scores.flatten(0)
            loss_boxiou = F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')

        num_boxes = src_boxes.shape[0] if self.ota else num_boxes
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes),box_ops.box_cxcywh_to_xyxy(target_boxes))
        # loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        #     box_ops.box_cxcywh_to_xyxy(src_boxes),
        #     box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        if 'pred_boxious' in outputs:
            losses['loss_boxiou'] = loss_boxiou
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"] # list[tensor]: bs x [1, num_inst, num_frames, H/4, W/4]
        bs = len(targets)
        # src_masks: bs x [1, num_inst, num_frames, H/4, W/4] or [bs, num_inst, num_frames, H/4, W/4]
        if type(src_masks) == list:
            src_masks = torch.cat(src_masks, dim=1)[0]  # [num_all_inst, num_frames, H/4, W/4]
        if src_masks.ndim == 0:
            # no mask label (only box label)
            losses = {}
            losses['loss_mask'] = src_masks * 0.0
            losses['loss_dice'] = src_masks * 0.0
            return losses

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks = self.get_target_masks(targets, src_masks)
        num_frames = src_masks.shape[1]
        # # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
        #                         mode="bilinear", align_corners=False)
        target_masks = target_masks.reshape(bs, -1, num_frames, target_masks.shape[-2], target_masks.shape[-1])
        target_masks = target_masks[tgt_idx] # [num_all_inst, num_frames, H/4, W/4]

        num_boxes = src_masks.shape[0] if self.ota else num_boxes

        if len(target_masks) == 0: # no gt
            losses = {}
            losses['loss_mask'] = src_masks.sum() * 0.0
            losses['loss_dice'] = src_masks.sum() * 0.0
            return losses
        
        src_masks = src_masks.flatten(1)  
        target_masks = target_masks.flatten(1)
        # src_masks/target_masks: [n_targets, num_frames* H * W]

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_masks_boxinst(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        self._iter += 1
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"] # list[tensor]: bs x [1, num_inst, num_frames, H/4, W/4]
        bs = len(targets)
        # src_masks: bs x [1, num_inst, num_frames, H/4, W/4] or [bs, num_inst, num_frames, H/4, W/4]
        if type(src_masks) == list:
            src_masks = torch.cat(src_masks, dim=1)[0]  # [num_all_inst, num_frames, H/4, W/4]
        if src_masks.ndim == 0:
            # no mask label (only box label)
            losses = {}
            losses['loss_prj'] = src_masks * 0.0
            losses['loss_pairwise'] = src_masks * 0.0
            return losses

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        # pick part of samples to compute loss because BoxInst consumes more memory
        if len(tgt_idx[0]) > self.boxinst_topk:
            keep_indexs = random.sample(range(len(tgt_idx[0])), self.boxinst_topk)
            src_idx = tuple([src_idx[0][keep_indexs], src_idx[1][keep_indexs]])
            tgt_idx = tuple([tgt_idx[0][keep_indexs], tgt_idx[1][keep_indexs]])
            src_masks = src_masks[keep_indexs]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks = self.get_target_masks(targets, src_masks)
        num_frames = src_masks.shape[1]
        # # upsample predictions to the target size
        # src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
        #                         mode="bilinear", align_corners=False)
        target_masks = target_masks.reshape(bs, -1, num_frames, target_masks.shape[-2], target_masks.shape[-1])
        target_masks = target_masks[tgt_idx] # [num_all_inst, num_frames, H/4, W/4]

        # num_boxes = src_masks.shape[0] if self.ota else num_boxes

        if len(target_masks) == 0: # no gt
            losses = {}
            losses['loss_prj'] = src_masks.sum() * 0.0
            losses['loss_pairwise'] = src_masks.sum() * 0.0
            return losses
        
        # box-supervised BoxInst losses
        mask_scores = src_masks.sigmoid()

        image_color_similarity_list = []
        for (batch_idx, target_idx) in zip(tgt_idx[0], tgt_idx[1]):
            image_color_similarity_list.append(targets[batch_idx]["image_color_similarity"][target_idx])
        image_color_similarity = torch.stack(image_color_similarity_list, dim=0).to(dtype=mask_scores.dtype) # (N, 8, H//4, W//4)

        loss_prj_term = compute_project_term(mask_scores, target_masks)

        pairwise_losses = compute_pairwise_term(
            src_masks, self.pairwise_size,
            self.pairwise_dilation
        )

        weights = (image_color_similarity >= self.pairwise_color_thresh).float() * target_masks.float() # (N, 8, H//4, W//4)
        loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

        warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
        loss_pairwise = loss_pairwise * warmup_factor

        losses = {
            "loss_prj": loss_prj_term,
            "loss_pairwise": loss_pairwise,
        }

        return losses

    def loss_reid(self, outputs, targets, indices, num_boxes):

        qd_items = outputs['pred_qd']
        contras_loss = 0
        aux_loss = 0
        if len(qd_items) == 0:
            losses = {'loss_reid': outputs['reid_params']*0,
                   'loss_reid_aux':  outputs['reid_params']*0}
            return losses
        for qd_item in qd_items:
            pred = qd_item['contrast'].permute(1,0)
            label = qd_item['label'].unsqueeze(0)
            # contrastive loss
            pos_inds = (label == 1)
            neg_inds = (label == 0)
            pred_pos = pred * pos_inds.float()
            pred_neg = pred * neg_inds.float()
            # use -inf to mask out unwanted elements.
            pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
            pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

            _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
            _neg_expand = pred_neg.repeat(1, pred.shape[1])
            # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
            x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0) 
            contras_loss += torch.logsumexp(x, dim=1)

            aux_pred = qd_item['aux_consin'].permute(1,0)
            aux_label = qd_item['aux_label'].unsqueeze(0)

            aux_loss += (torch.abs(aux_pred - aux_label)**2).mean()


        losses = {'loss_reid': contras_loss.sum()/len(qd_items),
                   'loss_reid_aux':  aux_loss/len(qd_items) }

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labelsVL': self.loss_labelsVL,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'reid': self.loss_reid,
            'masks_boxinst': self.loss_masks_boxinst,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, indices_list):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices_list[-1], num_boxes, **kwargs))

        if self.use_batch_loss:
            batch_losses = self.loss_VL_batch(outputs, targets, indices_list[-1], num_boxes)
            losses.update(batch_losses)
        if self.use_siglip_loss:
            sigmod_losses = self.sigmod_loss(outputs, targets, indices_list[-1], num_boxes)
            losses.update(sigmod_losses)
        if self.use_contrastive_focal_loss:
            contrastive_focal_losses = self.contrastive_focal_loss(outputs, targets, indices_list[-1], num_boxes)
            losses.update(contrastive_focal_losses)
        if self.use_clip_pool_head:
            clip_pool_losses = self.loss_text_image(outputs, targets, indices_list[-1], num_boxes)
            losses.update(clip_pool_losses)
        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                indices = indices_list[i]
                for loss in self.losses:
                    if loss == 'reid':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    if self.use_batch_loss:
                        batch_losses = self.loss_VL_batch(aux_outputs, targets, indices, num_boxes)
                        l_dict.update(batch_losses)
                    if self.use_siglip_loss:
                        sigmod_losses = self.sigmod_loss(aux_outputs, targets, indices, num_boxes)
                        l_dict.update(sigmod_losses)
                    if self.use_contrastive_focal_loss:
                        contrastive_focal_losses = self.contrastive_focal_loss(aux_outputs, targets, indices, num_boxes)
                        l_dict.update(contrastive_focal_losses)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels']) # set all labels to 0 (binary classification)
                if self.still_cls_for_encoder:
                    if "positive_map" in bt:
                        num_targets = len(bt["positive_map"])
                        bt["positive_map"] = torch.ones((num_targets, 1), dtype=torch.bool, device=bt["positive_map"].device)
                        enc_outputs['text_masks'] = None
            # if self.ota:
            #     indices, matched_ids = self.matcher.forward_ota(enc_outputs, bin_targets)
            # else:
            # We do not use OTA for the first stage
            indices = self.matcher.forward(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', 'reid', "masks_boxinst"]:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                # if self.use_batch_loss:
                #     batch_losses = self.loss_VL_batch(enc_outputs, bin_targets, indices, num_boxes)
                #     l_dict.update(batch_losses)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def get_target_masks(self, targets, src_masks):
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets], 
                                                            size_divisibility=32,
                                                            split=False).decompose()
        target_masks = target_masks.to(src_masks)
        # downsample ground truth masks with ratio mask_out_stride
        if self.mask_out_stride != 1:
            start = int(self.mask_out_stride // 2)
            im_h, im_w = target_masks.shape[-2:]
            target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
            assert target_masks.size(2) * self.mask_out_stride == im_h
            assert target_masks.size(3) * self.mask_out_stride == im_w
        return target_masks


class DINOCriterion(SetCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def forward(self, outputs, targets, indices_list, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = super(DINOCriterion, self).forward(outputs, targets, indices_list)
        # import pdb;pdb.set_trace()
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses

        aux_num = 0
        if "aux_outputs" in outputs:
            aux_num = len(outputs["aux_outputs"])
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)
        losses.update(dn_losses)

        return losses

    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
        """
        compute dn loss in criterion
        Args:
            dn_metas: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
            focal_alpha:  for focal loss
        """
        losses = {}
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = (
                dn_metas["output_known_lbs_bboxes"],
                dn_metas["dn_num"],
                dn_metas["single_padding"],
            )
            dn_idx = []
            # loop over batchsize
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["labels"])).long().cuda()
                    t = t.unsqueeze(0).repeat(dn_num, 1) # shape: (dn_num, n)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(dn_num)) * single_padding
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                if loss not in ['labelsVL', 'boxes']:
                    continue
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False}
                l_dict.update(
                    self.get_loss(
                        loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num, **kwargs
                    )
                )

            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")

        for i in range(aux_num):
            # dn aux loss
            l_dict = {}
            if dn_metas and "output_known_lbs_bboxes" in dn_metas:
                output_known_lbs_bboxes_aux = output_known_lbs_bboxes["aux_outputs"][i]
                for loss in self.losses:
                    if loss not in ['labelsVL', 'boxes']:
                        continue
                    kwargs = {}
                    if "labels" in loss:
                        kwargs = {"log": False}
                    l_dict.update(
                        self.get_loss(
                            loss,
                            output_known_lbs_bboxes_aux,
                            targets,
                            dn_idx,
                            num_boxes * dn_num,
                            **kwargs,
                        )
                    )
                l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
            else:
                l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)

    return unfolded_x

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4

    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]

def compute_box_iou(inputs, targets):
    """Compute pairwise iou between inputs, targets
    Both have the shape of [N, 4] and xyxy format
    """
    area1 = box_ops.box_area(inputs)
    area2 = box_ops.box_area(targets)

    lt = torch.max(inputs[:, None, :2], targets[:, :2])  # [N,M,2]
    rb = torch.min(inputs[:, None, 2:], targets[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)        # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    iou = torch.diag(iou)
    return iou

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

