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

class RetrievalEvaluator(DatasetEvaluator):
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
        self.use_iou_branch = cfg.MODEL.USE_IOU_BRANCH

    

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
            pred = output._fields

            image_id = input["image_id"]
            image_path = input["file_name"]
            result['gallery_features'] = pred["vis_features"].squeeze(0).to("cpu")
            result['clip_image_features'] = pred["clip_image_features"].to("cpu")
            result['image_id'] = image_id
            result['image_path'] = image_path
            result['text_features'] = pred["lang_features"].to("cpu")
            result['clip_text_features'] = pred["clip_text_features"].to("cpu")
            if self.use_iou_branch:
                result['bbox_threshold'] = pred["iou_thresh"].squeeze(0).unsqueeze(1).to("cpu")
            
            result['retrieval_text'] = input['expressions']

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

        # 1. 合并所有的 gallery_features 和对应的 image_id
        all_gallery_features = []
        all_gallery_image_ids = []
        all_gallery_image_paths = []
        if self.use_iou_branch:
            all_bbox_thresholds = []
        all_text_features = []
        all_query_image_ids = []
        all_query_image_paths = []
        all_query_text = []

        all_clip_image_features = []
        all_clip_text_features = []

        
        for result in results:
            all_gallery_features.append(result['gallery_features'])
            all_gallery_image_ids.append(result['image_id'])
            all_gallery_image_paths.append(result['image_path']) 
            if self.use_iou_branch:
                all_bbox_thresholds.append(result['bbox_threshold'])          

            all_text_features.append(result['text_features'])  # 提取所有text_features
            all_query_image_ids.append(result['image_id'])  # 提取每个查询的真实image_id
            all_query_image_paths.append(result['image_path'])
            all_query_text.append(result['retrieval_text'])  # 修正扩展方式

            all_clip_image_features.append(result['clip_image_features'])
            all_clip_text_features.append(result['clip_text_features'])

        
        # 将列表合并为 numpy 数组，方便计算
        # all_gallery_features = torch.vstack(all_gallery_features)  # 形状：(N, D)，N 为总数，D 为特征维度
        # all_gallery_image_ids = np.hstack(all_gallery_image_ids)  # 形状：(N,)
        # if self.use_iou_branch:
        #     all_bbox_thresholds = torch.vstack(all_bbox_thresholds)
        # all_gallery_image_paths = np.hstack(all_gallery_image_paths)  # 修正变量名
        all_text_features = torch.vstack(all_text_features)  # 形状：(M, D)，M 为查询数
        all_query_image_ids = np.array(all_query_image_ids)  # 形状：(M,)
        all_query_image_paths = np.array(all_query_image_paths)
        # all_query_text = np.array(all_query_text)  # 形状：(M,)
        all_clip_image_features = torch.vstack(all_clip_image_features)
        all_clip_text_features = torch.vstack(all_clip_text_features)


        num_queries = len(all_text_features)
        num_images = len(all_gallery_features)
        final_similarities = np.zeros((num_queries, num_images))

        # 遍历每张图片，计算文本与该图片的所有特征的相似度，并取最大值
        for img_idx, gallery_features in enumerate(all_gallery_features):  
            # gallery_features 形状: (K, D)，K 是该图像的多个特征数量
            gallery_features = gallery_features.to(all_text_features.device)  # 确保设备一致
            
            # 计算该图像的所有特征与所有文本的相似度 (K, M)
            similarity_matrix = gallery_features @ all_text_features.T  # 形状: (K, M)

            # 确保相似度矩阵非空且维度正确
            if similarity_matrix.numel() == 0:
                print(f"Warning: Similarity matrix for image {img_idx} is empty!")
                continue  # 如果空矩阵，跳过此图

            if similarity_matrix.shape[0] == 0 or similarity_matrix.shape[1] == 0:
                print(f"Warning: Invalid similarity matrix shape {similarity_matrix.shape} for image {img_idx}.")
                continue  # 如果相似度矩阵维度无效，跳过
            
            # # 取 Top-5 最大相似度，并计算平均值
            # top5_similarities = similarity_matrix.topk(k=min(5, similarity_matrix.shape[0]), dim=0)[0]  # (5, M)
            # avg_top5_similarity = top5_similarities.mean(dim=0).cpu().numpy()  # (M,)
            avg_similarity = similarity_matrix.max(dim=0)[0].cpu().numpy()

            # 存入最终相似度矩阵 (M, N)
            final_similarities[:, img_idx] = avg_similarity


        clip_similarity_matrix = all_clip_text_features @ all_clip_image_features.T
        
        #Text->Images 
        ranks = np.zeros(final_similarities.shape[0])

        for index,score in enumerate(final_similarities):
            gt_image_id = all_query_image_ids[index]
            gt_index = np.where(all_gallery_image_ids == gt_image_id)[0][0]

            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == gt_index)[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)    

        clip_ranks = np.zeros(clip_similarity_matrix.shape[0])

        for index,score in enumerate(clip_similarity_matrix.numpy()):
            gt_image_id = index
            inds = np.argsort(score)[::-1]
            clip_ranks[index] = np.where(inds == gt_index)[0][0]

        # Compute metrics
        clip_ir1 = 100.0 * len(np.where(clip_ranks < 1)[0]) / len(clip_ranks)
        clip_ir5 = 100.0 * len(np.where(clip_ranks < 5)[0]) / len(clip_ranks)
        clip_ir100 = 100.0 * len(np.where(clip_ranks < 100)[0]) / len(clip_ranks)   

        
        metrics = {
            "tir1": ir1,
            "tir5": ir5,
            "tir10": ir10,
            "clip_ir1": clip_ir1,
            "clip_ir5": clip_ir5,
            "clip_ir100": clip_ir100,

        }

        return metrics