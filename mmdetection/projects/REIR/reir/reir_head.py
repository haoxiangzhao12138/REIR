import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.dense_heads.detr_head import DETRHead
import math
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmengine.structures import InstanceData

class VL_Align(nn.Module):
    def __init__(self, prior_prob, log_scale, max_num,clamp=True):
        super().__init__()
        self.clamp = clamp
        self.max_num = max_num

        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # dot product soft token head
        self.dot_product_projection_image = nn.Identity()
        self.dot_product_projection_text = nn.Linear(1024, 256, bias=True) # 1024 -> 256
        self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(1024), requires_grad=True) # (1024,)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True) # size (1,)
    
    def forward(self, x, embeddings):
        """
        x: visual features (bs, num_query, 256)
        embedding: language features (bs, L, 1024)
        """
        # L_max = max([embedding.shape[0] for embedding in embeddings])
        L_max = self.max_num
        bs = len(embeddings)
        
        # 填充 embeddings 以匹配最大长度 L_max，并创建掩码
        padded_embeddings = torch.zeros(bs, L_max, 1024, device=x.device)
        mask = torch.zeros(bs, L_max, dtype=torch.bool, device=x.device)
        for i, embedding in enumerate(embeddings):
            L_i = embedding.shape[0]
            padded_embeddings[i, :L_i] = embedding
            mask[i, :L_i] = 1  # 有效部分设置为 1
        # 归一化
        normalized_embeddings = F.normalize(padded_embeddings, p=2, dim=-1)  # (bs, L_max, 1024)
        
        # 投影到 256 维
        dot_product_proj_tokens = self.dot_product_projection_text(normalized_embeddings / 2.0)  # (bs, L_max, 256)

        # 加上偏置项
        dot_product_proj_tokens_bias = (padded_embeddings @ self.bias_lang) + self.bias0  # (bs, L_max)
        
        # 图像查询的投影
        dot_product_proj_queries = self.dot_product_projection_image(x)  # (bs, num_query, 256)
        A = dot_product_proj_queries.shape[1]  # num_query

        # 扩展偏置项以匹配查询数目
        bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)  # (bs, num_query, L_max)

        # 计算点积得分，并应用掩码
        dot_product_logit = (dot_product_proj_queries @ dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp() + bias  # (bs, num_query, L_max)
        
        # 可选的值范围截断
        if self.clamp:
            dot_product_logit = torch.clamp(dot_product_logit, max=50000)
            dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
        
        # 应用掩码，填充位置的得分设为无穷小
        dot_product_logit = dot_product_logit.masked_fill(~mask.unsqueeze(1), float('-inf'))
            
        return dot_product_logit
    
    def get_text_features(self, embeddings):
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)  # (bs, L_max, 1024)
        
        # 投影到 256 维
        return self.dot_product_projection_text(normalized_embeddings / 2.0)
        
        # # norm
        # embedding = F.normalize(embedding, p=2, dim=-1) # (bs, L, 768) L is maximum sentence length
        # dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0) # 768 -> 256
        # dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0 # (bs, L, 768) x (768, ) + (1, ) -> (bs, L)

        # dot_product_proj_queries = self.dot_product_projection_image(x) # (bs, num_query, 256)
        # A = dot_product_proj_queries.shape[1] # num_query
        # bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1) # (bs, num_query, L)

        # dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias # (bs, num_query, 256) x (bs, 256, L) -> (bs, num_query, L)
        # if self.clamp:
        #     dot_product_logit = torch.clamp(dot_product_logit, max=50000)
        #     dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
        # return dot_product_logit



@MODELS.register_module()
class ReirHead(DETRHead):
    r"""
    Args:
        share_pred_layer (bool): Whether to share parameters for all the
            prediction layers. Defaults to `False`.
        num_pred_layer (int): The number of the prediction layers.
            Defaults to 6.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
    """

    def __init__(self,
                 *args,
                 share_pred_layer: bool = False,
                 num_pred_layer: int = 6,
                 as_two_stage: bool = False,
                 prior_prob: float = 0.01,
                 log_scale: float = 1.0,
                 **kwargs) -> None:
        self.share_pred_layer = share_pred_layer
        self.num_pred_layer = num_pred_layer
        self.as_two_stage = as_two_stage
        self.prior_prob = prior_prob
        self.log_scale = log_scale
        self.max_num = kwargs.get('num_classes', 10)

        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        cls_branch = VL_Align(self.prior_prob, self.log_scale, self.max_num)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        if self.share_pred_layer:
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])

        else:
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])
            # self.cls_branches = nn.ModuleList([
            #     copy.deepcopy(cls_branch) for _ in range(self.num_pred_layer)
            # ])
        self.cls_branches = nn.ModuleList(
            [cls_branch for _ in range(self.num_pred_layer)]
        )

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor],
                text_feats) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state, text_feats)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords# (cx, cy, w, h)

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, text_feats) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passed in,
                otherwise it would be `None`.
            enc_outputs_coord (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references, text_feats)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passes in,
                otherwise, it would be `None`.
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = super().loss_by_feat(all_layers_cls_scores,
                                         all_layers_bbox_preds,
                                         batch_gt_instances, batch_img_metas,
                                         batch_gt_instances_ignore)

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            proposal_gt_instances = copy.deepcopy(batch_gt_instances)
            for i in range(len(proposal_gt_instances)):
                proposal_gt_instances[i].labels = torch.zeros_like(
                    proposal_gt_instances[i].labels)
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=proposal_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
        return loss_dict

    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                batch_data_samples: SampleList,
                text_feats,
                rescale: bool = True,
                ) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references, text_feats)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Default `False`.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list
    
    def ref_predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                batch_data_samples: SampleList,
                text_feats,
                rescale: bool = True,
                ) -> InstanceList:

        all_layers_outputs_classes, all_layers_outputs_coord = self(hidden_states, references, text_feats)
        cls_scores = all_layers_outputs_classes[-1]
        bbox_preds = all_layers_outputs_coord[-1]

        result_list = []
        for img_id in range(len(batch_data_samples)):
            text_num = len(batch_data_samples[img_id].text)
            img_meta = batch_data_samples[img_id].metainfo
            img_shape = img_meta['img_shape']

            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            similarity = cls_score[:, :text_num].sigmoid()
            scores, bbox_index = torch.max(similarity, dim=0)
            bbox_pred = bbox_pred[bbox_index]

            det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
            det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
            det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
            det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
            det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

            if rescale:
                assert img_meta.get('scale_factor') is not None
                det_bboxes /= det_bboxes.new_tensor(
                    img_meta['scale_factor']).repeat((1, 2))

            results = InstanceData()
            results.bboxes = det_bboxes
            results.scores = scores
            result_list.append(results)
        
        return result_list, all_layers_outputs_coord


        #     img_meta = batch_img_metas[img_id]
        #     results = self._predict_by_feat_single(cls_score, bbox_pred,
        #                                            img_meta, rescale)
        #     result_list.append(results)



        # return predictions
