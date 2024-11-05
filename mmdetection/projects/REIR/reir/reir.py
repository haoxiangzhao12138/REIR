# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmengine.model import xavier_init
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmdet.models.layers import (DeformableDetrTransformerDecoder,
                      DeformableDetrTransformerEncoder, SinePositionalEncoding)
from mmdet.models.detectors.base_detr import DetectionTransformer
from mmdet.models.detectors.deformable_detr import DeformableDETR

from .utils import COCO_CLASS
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
@MODELS.register_module()
class REIR(DeformableDETR):
    r"""
    Args:
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding box head module. Defaults to None.
        with_box_refine (bool, optional): Whether to refine the references
            in the decoder. Defaults to `False`.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
        num_feature_levels (int, optional): Number of feature levels.
            Defaults to 4.  
    """

    def __init__(self,
                 *args,
                 dataset_name='coco',
                 freeze_text_encoder=True,
                 **kwargs) -> None:
        super(REIR, self).__init__(*args, **kwargs)

        self.dataset_name = dataset_name
        self.freeze_text_encoder = freeze_text_encoder

        self.clip_text_model = self.backbone.base_model.adaptors['clip']
        # in_dim = self.clip_text_model.oc_model.ln_final.weight.shape[0]
        # out_dim = 256
        # projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
        #                           nn.Dropout(0.0), ]
        # self.text_feature_projection = nn.ModuleList([nn.Sequential(*projection_layers)])
    
    def get_text_feats(self, batch_data_samples):
        b = len(batch_data_samples)
        if 'text' in batch_data_samples[0]:
            text_feats = []
            for data_sample in batch_data_samples:
                text = data_sample.text
                tokenized_texts = self.clip_text_model.tokenizer(text).cuda()
                text_feats.append(self.clip_text_model.encode_text(tokenized_texts))
            # text_feats = torch.stack(text_feats)
        else:
            if self.dataset_name == 'coco':
                tokenized_class = self.clip_text_model.tokenizer(COCO_CLASS).cuda()
            text_feats = self.clip_text_model.encode_text(tokenized_class)
            text_feats = [text_feats] * b
        # for module in self.text_feature_projection:
        #     text_feats = module(text_feats)

        return text_feats

    def loss(self, batch_inputs, batch_data_samples):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        # for idx, (name, param) in enumerate(self.named_parameters()):
        #     print(f"Index: {idx}, Parameter Name: {name}, Requires Grad: {param.requires_grad}")
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(name)

        # if 'tokenized_text' in batch_inputs:
        #     tokenized_text = batch_inputs['tokenized_text']
        # else:
        #     tokenized_text = None
        # image_input = batch_inputs['image_input']
        img_feats = self.extract_feat(batch_inputs)
        text_feats = self.get_text_feats(batch_data_samples)
        self.text_feats = text_feats
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        losses = self.bbox_head.loss(**head_inputs_dict, batch_data_samples=batch_data_samples, text_feats=text_feats)
        return losses
    
    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # rescale=False
        img_feats = self.extract_feat(batch_inputs)
        text_feats = self.get_text_feats(batch_data_samples)
        self.text_feats = text_feats
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        if 'text' in batch_data_samples[0]:
            results_list, all_layers_outputs_coord = self.bbox_head.ref_predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples,
                text_feats=text_feats)

            for i, data_sample in enumerate(batch_data_samples):
                bbox_pred = all_layers_outputs_coord[-1][i]
                img_h, img_w = batch_data_samples[i].metainfo['img_shape']
                factor = bbox_pred.new_tensor([img_w, img_h, img_w,img_h]).unsqueeze(0)
                bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
                bbox_pred = bbox_pred * factor

                text_num = len(data_sample.text)
                data_sample.text_feature = self.text_feats[i][:text_num]
                data_sample.gallery_features = head_inputs_dict['hidden_states'][-1][i]
                data_sample.gallery_bboxes = bbox_pred
                data_sample.text_features = self.bbox_head.cls_branches[-1].get_text_features(self.text_feats[i])
        else:
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples,
                text_feats=text_feats)
                
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        # TODO 加上retrieval branch的结果



        return results
    
    def pre_decoder(self, memory: Tensor, memory_mask: Tensor,
                    spatial_shapes: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). It will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
                It will only be used when `as_two_stage` is `True`.

        Returns:
            tuple[dict, dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory', and `reference_points`. The reference_points of
              decoder input here are 4D boxes when `as_two_stage` is `True`,
              otherwise 2D points, although it has `points` in its name.
              The reference_points in encoder is always 2D points.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `enc_outputs_class` and
              `enc_outputs_coord`. They are both `None` when 'as_two_stage'
              is `False`. The dict is empty when `self.training` is `False`.
        """
        batch_size, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.bbox_head.cls_branches[
                self.decoder.num_layers](
                    output_memory, self.text_feats)
            enc_outputs_coord_unact = self.bbox_head.reg_branches[
                self.decoder.num_layers](output_memory) + output_proposals
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            # We only use the first channel in enc_outputs_class as foreground,
            # the other (num_classes - 1) channels are actually not used.
            # Its targets are set to be 0s, which indicates the first
            # class (foreground) because we use [0, num_classes - 1] to
            # indicate class labels, background class is indicated by
            # num_classes (similar convention in RPN).
            # See https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/deformable_detr_head.py#L241 # noqa
            # This follows the official implementation of Deformable DETR.
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], self.num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            pos_trans_out = self.pos_trans_fc(
                self.get_proposal_pos_embed(topk_coords_unact))
            pos_trans_out = self.pos_trans_norm(pos_trans_out)
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            enc_outputs_class, enc_outputs_coord = None, None
            query_embed = self.query_embedding.weight
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
            query = query.unsqueeze(0).expand(batch_size, -1, -1)
            reference_points = self.reference_points_fc(query_pos).sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points)
        head_inputs_dict = dict(
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

