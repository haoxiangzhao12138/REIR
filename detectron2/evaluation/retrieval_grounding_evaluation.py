# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from .evaluator import DatasetEvaluator
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import textwrap

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval
from detectron2.evaluation.refcocoeval import RefCOCOeval

class RetrievalGroundingEvaluator(DatasetEvaluator):
    """REIR Metric."""

    def __init__(self, 
                 cfg):
        self.retrieval = cfg.MODEL.RETRIEVAL
        self.topk_list = [1,5,10]
        self._distributed=True
        self.log_scale = 0.9364
        self.use_bias = cfg.MODEL.USE_BIAS
        self.visualize = cfg.MODEL.VISUALIZE
        self.use_multithreading = cfg.MODEL.USE_MULTITHREADING
        self.mask_on = cfg.MODEL.MASK_ON

    def compute_iou_batch_gpu(self, bboxes1, bboxes2):
        """GPU 上计算多个边界框的 IOU"""
        bboxes1 = bboxes1.half().unsqueeze(1)  # [N, 1, 4]
        bboxes2 = bboxes2.half().unsqueeze(0)  # [1, M, 4]
        
        inter_min = torch.max(bboxes1[:, :, :2], bboxes2[:, :, :2])
        inter_max = torch.min(bboxes1[:, :, 2:], bboxes2[:, :, 2:])
        inter_area = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)

        area1 = torch.prod(bboxes1[:, :, 2:] - bboxes1[:, :, :2], dim=2)
        area2 = torch.prod(bboxes2[:, :, 2:] - bboxes2[:, :, :2], dim=2)
        union_area = area1 + area2 - inter_area
        return (inter_area / union_area).float()  # 转回 float32   
     
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
    

    def process_query_batch(self, batch_indices, similarity_matrix, all_query_image_ids, all_query_image_paths, 
                            all_query_text, all_gallery_image_ids, all_gallery_image_paths, 
                            all_gt_bboxes, all_gallery_bboxes, topk_list):
        """批量处理多个查询并进行可视化。

        参数:
            batch_indices (list 或 array): 当前批次中查询的索引。
            similarity_matrix (np.ndarray): 查询与图库之间的相似度矩阵。
            all_query_image_ids (list): 所有查询图像的ID。
            all_query_image_paths (list): 所有查询图像的文件路径。
            all_query_text (list): 所有查询图像的文本注释。
            all_gallery_image_ids (list): 所有图库图像的ID。
            all_gallery_image_paths (list): 所有图库图像的文件路径。
            all_gt_bboxes (list): 查询图像的地面真值边界框。
            all_gallery_bboxes (list): 库图图像的边界框。
            topk_list (list): 需要评估的k值列表（如 [1, 5, 10]）。

        返回:
            dict: 包含每个k值的匹配数、找到数和总数的字典。
        """
        # 初始化结果字典
        results = {k: {'matches': 0, 'found': 0, 'total': 0} for k in topk_list}
        
        # 提取当前批次的相似度并按降序排序
        similarities = similarity_matrix[batch_indices]  # [batch_size, num_gallery]
        sorted_indices = np.argsort(similarities, axis=1)[:, ::-1]  # 降序排序索引

        # 获取当前批次的查询相关数据
        query_image_ids = [all_query_image_ids[idx] for idx in batch_indices]
        query_image_paths = [all_query_image_paths[idx] for idx in batch_indices]
        query_texts = [all_query_text[idx] for idx in batch_indices]
        gt_bboxes = [all_gt_bboxes[idx] for idx in batch_indices]

        for query_idx, (query_id, query_path, query_text, gt_bbox) in enumerate(zip(query_image_ids, 
                                                                                    query_image_paths, 
                                                                                    query_texts, 
                                                                                    gt_bboxes)):
            for k in topk_list:
                top_k_indices = sorted_indices[query_idx, :k]
                top_k_gallery_paths = [all_gallery_image_paths[idx] for idx in top_k_indices]
                top_k_similarities = [similarities[query_idx, idx] for idx in top_k_indices]
                
                # 检查查询ID是否在前k个图库ID中
                if query_path in top_k_gallery_paths:
                    results[k]['matches'] += 1
                    for idx in top_k_indices:
                        gallery_bbox = all_gallery_bboxes[idx]
                        # 如果边界框是张量，则转换为numpy数组
                        if isinstance(gallery_bbox, torch.Tensor):
                            gallery_bbox = gallery_bbox.cpu().numpy()
                        # 计算IoU并判断是否满足阈值
                        if self.compute_iou(gt_bbox, gallery_bbox) >= 0.5:
                            results[k]['found'] += 1
                            break   
                
                results[k]['total'] += 1

            # 可视化部分
            if self.visualize:
                if query_idx % 50 == 0:
                    # 获取前5个图库图像的索引、ID、路径、边界框和相似度
                    top_5_indices = sorted_indices[query_idx, :5]
                    top_5_gallery_paths = [all_gallery_image_paths[idx] for idx in top_5_indices]
                    top_5_gallery_bboxes = [all_gallery_bboxes[idx] for idx in top_5_indices]
                    top_5_similarities = [similarities[query_idx, idx] for idx in top_5_indices]

                    # 加载并绘制查询图像
                    try:
                        query_image = Image.open(query_path).convert("RGB")
                    except Exception as e:
                        print(f"加载查询图像 {query_path} 时出错: {e}")
                        continue

                    draw_query = ImageDraw.Draw(query_image)
                    # 绘制地面真值边界框（绿色）
                    if isinstance(gt_bbox, torch.Tensor):
                        gt_bbox = gt_bbox.cpu().numpy()
                    draw_query.rectangle(list(gt_bbox), outline="green", width=2)
                    # 在边界框上方添加查询文本
                    try:
                        font = ImageFont.truetype("arial.ttf", 25)
                    except IOError:
                        # 退而求其次使用默认字体
                        font = ImageFont.load_default(size=30)

                    # 为避免过长的查询文本被截断，这里使用 textwrap.fill 做自动换行
                    wrapped_text = textwrap.fill(query_text, width=40)

                    # 固定到左上角：直接指定一个位置，例如 (10, 10)
                    text_position = (10, 10)

                    # 绘制多行文本
                    draw_query.multiline_text(
                        text_position, 
                        wrapped_text, 
                        fill="black", 
                        font=font, 
                        spacing=4
                    )


                    # 加载并绘制图库图像
                    gallery_images = []
                    for img_path, bbox, sim in zip(top_5_gallery_paths, top_5_gallery_bboxes, top_5_similarities):
                        try:
                            img = Image.open(img_path).convert("RGB")
                            draw_img = ImageDraw.Draw(img)
                            # 如果边界框是张量，则转换为numpy数组
                            if isinstance(bbox, torch.Tensor):
                                bbox = bbox.cpu().numpy()
                            # 绘制图库边界框（红色）
                            draw_img.rectangle(list(bbox), outline="red", width=5)
                            # 在边界框上方添加相似度分数
                            sim_text = f"{sim:.2f}"
                            sim_position = (bbox[0], bbox[1] - 10) if bbox[1] - 10 > 0 else (bbox[0], bbox[1] + 5)
                            draw_img.text(sim_position, sim_text, fill="black", font=font)
                            gallery_images.append(img)
                        except Exception as e:
                            print(f"加载图库图像 {img_path} 时出错: {e}")
                            # 如果加载失败，添加一张空白图像作为占位
                            blank_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
                            gallery_images.append(blank_img)

                    # 创建一个组合图像：左侧为查询图像，右侧为前5个图库图像
                    all_images = [query_image] + gallery_images
                    widths, heights = zip(*(i.size for i in all_images))
                    total_width = sum(widths)
                    max_height = max(heights)

                    composite_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
                    x_offset = 0
                    for img in all_images:
                        composite_image.paste(img, (x_offset, 0))
                        x_offset += img.size[0]

                    # 可选择将可视化结果保存到磁盘
                    save_dir = "visualizations"
                    os.makedirs(save_dir, exist_ok=True)
                    composite_image.save(os.path.join(save_dir, f'query_{query_id}.jpg'))

        return results

    
    
    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            result = dict()
            pred = output["instances"].to(torch.device("cpu"))

            if len(input['annotations']) != 0:
                x, y, w, h = input['annotations'][0]['bbox']
                x2 = x + w
                y2 = y + h
                gt_bbox_xyxy = [x, y, x2, y2]
                result['REC_gt'] = gt_bbox_xyxy
            else:
                result['REC_gt'] = [0, 0, 0, 0]
            
            image_id = input["image_id"]
            image_path = input["file_name"]
            result['gallery_features'] = pred.vis_features
            result['gallery_bboxes'] = pred.pred_boxes.tensor
            result['image_id'] = [image_id] * len(result['gallery_bboxes'])
            result['image_path'] = [image_path] * len(result['gallery_bboxes'])
            result['text_features'] = pred.lang_features[0]
            
            result['REC_text'] = input['expressions']
            if self.use_bias:
                result['bias'] = pred.bias.squeeze(0)
                result['scale'] = pred.scale

            self._predictions.append(result)


    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        # if self._output_dir:
        #     PathManager.mkdirs(self._output_dir)
        #     file_path = os.path.join(self._output_dir, "instances_predictions.pth")
        #     with PathManager.open(file_path, "wb") as f:
        #         torch.save(predictions, f)

        self._results = self.compute_metrics(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


    def compute_metrics(self, results: list):
        metrics = OrderedDict()

        # 1. 合并所有的 gallery_features 和对应的 image_id
        all_gallery_features = []
        all_gallery_bboxes = []
        all_gallery_image_ids = []
        all_gallery_image_paths = []
        all_text_features = []
        all_query_image_ids = []
        all_query_image_paths = []
        all_query_text = []
        all_gt_bboxes = []
        if self.use_bias:
            all_bias = []
            all_scale = []
        
        for result in results:
            if result['image_path'][0] not in all_query_image_paths:     
                all_gallery_features.append(result['gallery_features'])
                all_gallery_bboxes.append(result['gallery_bboxes'])
                all_gallery_image_ids.append(result['image_id'])
                all_gallery_image_paths.append(result['image_path'])           

            all_text_features.append(result['text_features'])  # 提取所有text_features
            num_text_features = len(result['text_features'])
            all_query_image_ids.extend([result['image_id'][0]] * num_text_features)  # 提取每个查询的真实image_id
            all_query_image_paths.extend([result['image_path'][0]] * num_text_features)
            all_query_text.extend([result['REC_text']] * num_text_features)  # 修正扩展方式
            all_gt_bboxes.extend([result['REC_gt']] * num_text_features)     # 修正扩展方式
            if self.use_bias:
                all_bias.append(result['bias'])
                all_scale.append(result['scale'])

        
        # 将列表合并为 numpy 数组，方便计算
        all_gallery_features = torch.vstack(all_gallery_features)  # 形状：(N, D)，N 为总数，D 为特征维度
        all_gallery_bboxes = torch.vstack(all_gallery_bboxes)  # 形状：(N, 4)
        all_gallery_image_ids = np.hstack(all_gallery_image_ids)  # 形状：(N,)
        all_gallery_image_paths = np.hstack(all_gallery_image_paths)  # 修正变量名
        all_text_features = torch.vstack(all_text_features)  # 形状：(M, D)，M 为查询数
        all_query_image_ids = np.array(all_query_image_ids)  # 形状：(M,)
        all_query_image_paths = np.array(all_query_image_paths)
        all_query_text = np.array(all_query_text)  # 形状：(M,)
        all_gt_bboxes = np.vstack(all_gt_bboxes)  # 形状：(M, 4)
        if self.use_bias:
            all_bias = torch.hstack(all_bias)
            all_scale = torch.hstack(all_scale)

        # 计算所有 text_features 和 gallery_features 之间的相似度 (M, N)
        chunk_size = 102400 
        num_chunks = (len(all_gallery_features) + chunk_size - 1) // chunk_size  # 计算需要的块数

        similarity_matrix_chunks = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(all_gallery_features))

            # 提取当前块
            gallery_chunk = all_gallery_features[start_idx:end_idx].cuda()
            text_features_cuda = all_text_features.cuda()  # 假设文本特征可以整体加载
            similarity_chunk = torch.matmul(gallery_chunk, text_features_cuda.transpose(-1, -2))
            similarity_matrix_chunks.append(similarity_chunk.cpu())  # 将结果移回 CPU 节省显存

        # 合并所有块
        similarity_matrix = torch.cat(similarity_matrix_chunks, dim=0)
        similarity_matrix = similarity_matrix.T.numpy()
        

        ranks = np.zeros(similarity_matrix.shape[0])
        valid_ranks = []
        for index,score in tqdm(enumerate(similarity_matrix)):
            gt_image_id = all_query_image_ids[index]
            gt_indices = np.where(all_gallery_image_ids == gt_image_id)[0]

            inds = np.argsort(score)[::-1]
            best_gt_index = np.min([np.where(inds == gt_idx)[0][0] for gt_idx in gt_indices])  # 取最小排名
            image_id_index = len(np.unique(all_gallery_image_ids[inds[:best_gt_index]]))
            ranks[index] = image_id_index

            # 计算 IoU
            pred_bbox = all_gallery_bboxes[gt_indices[np.argmin([np.where(inds == gt_idx)[0][0] for gt_idx in gt_indices])]]
            gt_bbox = all_gt_bboxes[index]

            if self.compute_iou(pred_bbox, gt_bbox) >= 0.5:
                valid_ranks.append(best_gt_index)
        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks) 

        # 计算 IoU 过滤后的 ir1, ir5, ir10
        valid_ranks = np.array(valid_ranks)
        ir1_iou = 100.0 * len(np.where(valid_ranks < 1)[0]) / len(ranks) 
        ir5_iou = 100.0 * len(np.where(valid_ranks < 5)[0]) / len(ranks) 
        ir10_iou = 100.0 * len(np.where(valid_ranks < 10)[0]) / len(ranks) 

        metrics = {
            'ir1': ir1,
            'ir5': ir5,
            'ir10': ir10,
            'ir1_iou': ir1_iou,
            'ir5_iou': ir5_iou,
            'ir10_iou': ir10_iou
        }


        # # 批量处理
        # batch_results = []
        # batch_size = 128

        # if self.use_multithreading:
        #     with ThreadPoolExecutor(max_workers=8) as executor:  # 控制线程数量（8 是一个合理的选择）
        #         futures = [
        #             executor.submit(
        #                 self.process_query_batch,
        #                 batch_indices=all_indices[start:start + batch_size],
        #                 similarity_matrix=similarity_matrix,
        #                 all_query_image_ids=all_query_image_ids,
        #                 all_query_image_paths=all_query_image_paths,
        #                 all_query_text=all_query_text,
        #                 all_gallery_image_ids=all_gallery_image_ids,
        #                 all_gallery_image_paths=all_gallery_image_paths,
        #                 all_gt_bboxes=all_gt_bboxes,
        #                 all_gallery_bboxes=all_gallery_bboxes,
        #                 topk_list=self.topk_list
        #             )
        #             for start in range(0, num_queries, batch_size)
        #         ]

        #         # 使用 as_completed 来正确显示进度条
        #         for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
        #             try:
        #                 batch_results.append(future.result())
        #             except Exception as e:
        #                 print(f"批处理过程中出现错误: {e}")
        # else:
        #     # 不使用多线程，顺序处理
        #     for start in tqdm(range(0, num_queries, batch_size), desc="Processing batches"):
        #         end = min(start + batch_size, num_queries)
        #         batch_result = self.process_query_batch(
        #             batch_indices=all_indices[start:end],
        #             similarity_matrix=similarity_matrix,
        #             all_query_image_ids=all_query_image_ids,
        #             all_query_image_paths=all_query_image_paths,
        #             all_query_text=all_query_text,
        #             all_gallery_image_ids=all_gallery_image_ids,
        #             all_gallery_image_paths=all_gallery_image_paths,
        #             all_gt_bboxes=all_gt_bboxes,
        #             all_gallery_bboxes=all_gallery_bboxes,
        #             topk_list=self.topk_list
        #         )
        #         batch_results.append(batch_result)

        # # 合并结果
        # metrics = {}
        # total_matches = {k: 0 for k in self.topk_list}
        # total_found = {k: 0 for k in self.topk_list}
        # total_queries = {k: 0 for k in self.topk_list}

        # for batch_result in batch_results:
        #     for k in self.topk_list:
        #         total_matches[k] += batch_result[k]['matches']
        #         total_found[k] += batch_result[k]['found']
        #         total_queries[k] += batch_result[k]['total']

        # # 计算最终指标
        # for k in self.topk_list:
        #     metrics[f'retrieval_recall@{k}'] = (
        #         total_found[k] / total_queries[k] if total_queries[k] > 0 else 0.0
        #     )
        #     metrics[f'retrieval_image_accuracy@{k}'] = (
        #         total_matches[k] / total_queries[k] if total_queries[k] > 0 else 0.0
        #     )

        return metrics