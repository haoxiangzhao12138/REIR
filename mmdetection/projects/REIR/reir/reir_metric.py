# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
from typing import Dict, List, Optional, Sequence, Union

import torch
from tqdm import tqdm
from mmengine.evaluator import BaseMetric
from sklearn.metrics.pairwise import cosine_similarity

from mmdet.registry import METRICS
from mmdet.evaluation.functional import bbox_overlaps
import numpy as np

@METRICS.register_module()
class ReirMetric(BaseMetric):
    """REIR Metric."""

    def __init__(self, 
                 metric: Union[str, List[str]] = 'bbox',
                 topk_list=[1, 5, 10],
                 **kwargs):
        super().__init__(**kwargs)
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.topk_list = topk_list

    def compute_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU值
        Args:
            bbox1 (np.ndarray): 预测的边界框，格式为 [x1, y1, x2, y2]
            bbox2 (np.ndarray): 真实的边界框，格式为 [x1, y1, x2, y2]
        Returns:
            float: IoU值
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # 计算交集面积
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算每个框的面积
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # 计算并集面积
        union_area = bbox1_area + bbox2_area - inter_area
        
        # IoU = 交集面积 / 并集面积
        iou = inter_area / union_area
        return iou

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            if 'REC' in self.metrics:
                pred = data_sample['pred_instances']
                result['REC_result'] = pred['bboxes'].cpu().numpy()
                result['REC_gt'] = data_sample['gt_instances']['bboxes'].cpu().numpy()
            
            if 'retrieval' in self.metrics:
                image_id = data_sample['img_id']
                result['gallery_features'] = data_sample['gallery_features'].cpu().numpy()
                result['gallery_bboxes'] = data_sample['gallery_bboxes'].cpu().numpy()
                result['image_id'] = [image_id] * len(result['gallery_bboxes'])
                result['text_features'] = data_sample['text_features'].cpu().numpy()
            
            if 'detection' in self.metrics:
                result['detection_result'] = data_sample['pred_instances']
                result['detection_gt'] = data_sample['gt_instances']
            
            self.results.append(result)


    def compute_metrics(self, results: list) -> dict:
        metrics = {}
        if 'REC' in self.metrics:
            correct_predictions = 0
            total_predictions = 0

            # 遍历每一个结果
            for result in results:
                rec_preds = result['REC_result']  # 预测的边界框
                rec_gts = result['REC_gt']        # 真实的边界框
                assert len(rec_gts)==len(rec_preds), 'The number of predictions and ground truths should be the same.'
                for i in range(len(rec_preds)):
                    iou = self.compute_iou(rec_preds[i], rec_gts[i])
                    if iou >= 0.5:
                        correct_predictions += 1  # IoU 大于 0.5 计为正确预测
                    total_predictions += 1

            # 计算准确率
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            metrics['REC_acc'] = accuracy

        if 'retrieval' in self.metrics:
            top_k_image_matches = {k: 0 for k in self.topk_list} 

            top_k_total_queries = {k: 0 for k in self.topk_list} 
            top_k_found = {k: 0 for k in self.topk_list} 

            # 1. 合并所有的 gallery_features 和对应的 image_id
            all_gallery_features = []
            all_gallery_bboxes = []
            all_gallery_image_ids = []
            all_text_features = []
            all_query_image_ids = []
            all_gt_bboxes = []
            
            for result in results:
                all_gallery_features.append(result['gallery_features'])
                all_gallery_bboxes.append(result['gallery_bboxes'])
                all_gallery_image_ids.append(result['image_id'])
                all_text_features.append(result['text_features'])  # 提取所有text_features
                all_query_image_ids.extend([result['image_id'][0]]*len(result['text_features']))  # 提取每个查询的真实image_id
                all_gt_bboxes.append(result['REC_gt'])             # 提取每个查询的真实物体bbox
    
            
            # 将列表合并为 numpy 数组，方便计算
            all_gallery_features = np.vstack(all_gallery_features)  # 形状：(N, D)，N 为总数，D 为特征维度
            all_gallery_bboxes = np.vstack(all_gallery_bboxes)  # 形状：(N, 4)
            all_gallery_image_ids = np.hstack(all_gallery_image_ids)  # 形状：(N,)
            all_text_features = np.vstack(all_text_features)  # 形状：(M, D)，M 为查询数
            all_query_image_ids = np.array(all_query_image_ids)  # 形状：(M,)
            all_gt_bboxes = np.vstack(all_gt_bboxes)  # 形状：(M, 4)

            # 计算所有 text_features 和 gallery_features 之间的相似度 (M, N)
            similarity_matrix = cosine_similarity(all_text_features, all_gallery_features)

            for query_idx in tqdm(range(similarity_matrix.shape[0])):
                similarities = similarity_matrix[query_idx]  # 当前查询与所有gallery的相似度
                query_image_id = all_query_image_ids[query_idx]  # 当前查询对应的真实图片ID
                gt_bbox = all_gt_bboxes[query_idx]  # 当前查询对应的真实物体边界框

                # 对相似度排序，获取排序后的索引
                sorted_indices = np.argsort(similarities)[::-1]  # 降序排列，索引从大到小


                for k in self.topk_list:  
                    top_k_indices = sorted_indices[:k]
                    for idx in top_k_indices:
                        if all_gallery_image_ids[idx] == query_image_id:
                            top_k_image_matches[k] += 1
                            iou = self.compute_iou(all_gallery_bboxes[idx], gt_bbox)
                            if iou >= 0.5:
                                top_k_found[k] += 1
                        top_k_total_queries[k] += 1

            metrics[f'retrieval_recall@{k}'] = top_k_found[k] / top_k_total_queries[k] if top_k_total_queries[k] > 0 else 0.0
            metrics[f'retrieval_image_accuracy@{k}'] = top_k_image_matches[k] / top_k_total_queries[k] if top_k_total_queries[k] > 0 else 0.0

        return metrics
