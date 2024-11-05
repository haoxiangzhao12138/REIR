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
from ..layers import (DeformableDetrTransformerDecoder,
                      DeformableDetrTransformerEncoder, SinePositionalEncoding)
from .base_detr import DetectionTransformer
from .deformable_detr import DeformableDETR



@MODELS.register_module()
class DeformableDETRWithTextRefocus(DeformableDETR):
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
                 retrieval_branch: OptConfigType = None,
                 **kwargs) -> None:
        super(DeformableDETRWithTextRefocus, self).__init__(*args, **kwargs)

        self.bbox_head.clip_model = self.backbone.base_model.adaptors['clip']
        self.bbox_head.clip_model.init_text_encoder()

        if retrieval_branch is not None:
            self.text_retrieval_branch = MODELS.build(retrieval_branch)
            self.text_retrieval_branch.clip_model = self.backbone.base_model.adaptors['clip']
            self.text_retrieval_branch.init_text_encoder()
    
    def get_text_query(self, batch_data_samples):
        return [sample.text[0] for sample in batch_data_samples]

    
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
        texts_feats = self.get_text_feats(batch_data_samples)
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        losses = dict()
        detect_losses, (all_layers_outputs_classes, all_layers_outputs_coords) = self.bbox_head.loss(**head_inputs_dict, batch_data_samples=batch_data_samples)
        
        losses.update(detect_losses)
        
        retrieval_losses = self.text_retrieval_branch.loss(head_inputs_dict['hidden_states'], all_layers_outputs_classes[-1], all_layers_outputs_coords[-1], 
                                                           batch_data_samples)

        losses.update(retrieval_losses)

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
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        det_results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        
        if 'text' in batch_data_samples[0]:
            retrieval_results, text_features = self.text_retrieval_branch.predict(head_inputs_dict['hidden_states'], det_results_list[0], batch_data_samples)

            for i in range(len(batch_data_samples)):
                batch_data_samples[i].ref_result = retrieval_results[i]
                batch_data_samples[i].gallery_features = head_inputs_dict['hidden_states'][-1][i]
                batch_data_samples[i].text_features = text_features[i]
        
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, det_results_list)
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

