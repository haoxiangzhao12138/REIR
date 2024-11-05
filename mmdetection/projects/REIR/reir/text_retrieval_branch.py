import torch.nn as nn
import torch
from mmdet.registry import MODELS
from mmdet.registry import MODELS, TASK_UTILS
from .utils import COCO_CLASS
from mmengine.model import BaseModel
from mmengine.config import ConfigDict
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from mmdet.datasets.objects365 import Objects365V2Dataset
import open_clip
from mmdet.structures import SampleList
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.attn_mask.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, ctx_cls_num, ctx_init, ctx_prompt_length, query_position, clip_model, tokenizer, with_text_prompt):
        super().__init__()
        n_cls = ctx_cls_num
        n_ctx = ctx_prompt_length
        self.dtype = clip_model.visual.conv1.weight.dtype
        ctx_dim = clip_model.text.ln_final.weight.shape[0]
        self.with_text_prompt = with_text_prompt
        if with_text_prompt:

            if ctx_init:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                prompt = tokenizer(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.text.token_embedding(prompt).type(self.dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                prompt_prefix = ctx_init

            else:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=self.dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)

            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")

            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized cls * length * dim

            prompts = [prompt_prefix]*n_cls

            tokenized_prompts = torch.cat([tokenizer(p) for p in prompts]) # n * 77
            with torch.no_grad():
                embedding = clip_model.text.token_embedding(tokenized_prompts).type(self.dtype)

            # These token vectors will be saved when in save_model(),
            # but they should be ignored in load_model() as we want to use
            # those computed using the current class names
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS

            self.n_cls = n_cls
            self.n_ctx = n_ctx
            self.query_position = query_position
            self.tokenizer = tokenizer
            self.token_embedding = clip_model.text.token_embedding
            self.prompt_prefix = prompt_prefix
        else:
            self.token_embedding = clip_model.text.token_embedding


    def forward(self, tokenized_text, text_clusters=None):
        if self.with_text_prompt:
            text_embedding = self.token_embedding(tokenized_text).type(self.dtype)

            ctx = self.ctx[0]
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix

            if self.query_position == "end":
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        text_embedding[:,1:,:],  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                raise ValueError

            return prompts
        else:
            text_embedding = self.token_embedding(tokenized_text).type(self.dtype)
            return text_embedding, tokenized_text


@MODELS.register_module()
class TextRetrievalBranch(BaseModule):
    def __init__(self, 
                #  name='ViT-H-14', 
                #  pretrained='/public/haoxiangzhao/weights/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin',
                 freeze_encoder=False,
                 loss_retrieval: ConfigType = dict(type='FocalLoss', loss_weight=2.0),
                 init_cfg: OptMultiConfig = None,
                 with_text_prompt=True,
                 with_ref_retrieval=True,
                 with_cata_retrieval=False,
                 ctx_cls_num = 2, 
                 ctx_init=None, 
                 ctx_prompt_length=8, 
                 query_position='end',
                 assigner=None,
                 kmeans_cluster_centers=None,
                 dataset_name='coco'
                 ):
        super().__init__(init_cfg=init_cfg)
        # self.clip, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(name, pretrained=pretrained, force_custom_text=True,)
        # self.tokenizer = open_clip.get_tokenizer(name)
        if kmeans_cluster_centers is not None:
            self.kmeans_cluster_centers = torch.load(kmeans_cluster_centers)
        # self.text_encoder = TextEncoder(self.clip.text)
        self.with_cata_retrieval = with_cata_retrieval
        # if self.with_cata_retrieval:
        #     self.coco_class = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        #     'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        #     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        #     'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        #     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        #     'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        #     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        #     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        #     'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        #     'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        #     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        #     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        #     'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        #     self.register_buffer('tokenized_class_name', self.tokenizer(self.coco_class))
        #     self.tokenized_class_name.requires_grad = False
        self.with_ref_retrieval = with_ref_retrieval
        self.with_text_prompt = with_text_prompt
        self.datasets_name = dataset_name
        
        # self.prompt_learner = PromptLearner(ctx_cls_num, ctx_init, ctx_prompt_length, query_position, self.clip, self.tokenizer, with_text_prompt)
        # in_dim = self.clip.text.ln_final.weight.shape[0]
        # out_dim = 256
        # projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
        #                           nn.Dropout(0.0), ]
        # self.text_feature_projection = nn.ModuleList([nn.Sequential(*projection_layers)])
        self.loss_retrieval = MODELS.build(loss_retrieval)

        # self.datasets_name = datasets
        # if self.datasets_name == 'objects365v2':
        #     class_name = Objects365V2Dataset.metainfo['classes']

        if assigner is not None:
            self.assigner = TASK_UTILS.build(assigner)

        # del self.clip

        # if freeze_encoder:
        #     self.freeze_clip()
    
    def init_text_encoder(self):
        self.text_encoder = TextEncoder(self.clip_model.oc_model)
        self.tokenizer = self.clip_model.tokenizer
        if self.datasets_name == 'coco':
            tokenized_class = self.clip_model.tokenizer(COCO_CLASS)
        
        self.register_buffer('tokenized_class', tokenized_class)
        self.tokenized_class.requires_grad = False

        in_dim = self.clip_model.oc_model.ln_final.weight.shape[0]
        out_dim = 256
        projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
                                  nn.Dropout(0.0), ]
        self.text_feature_projection = nn.ModuleList([nn.Sequential(*projection_layers)])
    
    def token_to_embedding(self, tokenized_text):
        text_embedding = self.clip_model.oc_model.token_embedding(tokenized_text)
        return text_embedding, tokenized_text


    

    def loss(self, hidden_states, final_layers_outputs_classes, final_layers_outputs_coords, batch_data_samples):
        # (cx, cy, w, h)
        b = len(batch_data_samples)
        if self.with_ref_retrieval:
            texts = []
            for d in batch_data_samples:
                texts.extend(d.text)
            tokenized_text = self.tokenizer(texts).cuda()
            prompts_embedding, tokenized_prompt = self.token_to_embedding(tokenized_text)

            features = self.text_encoder(prompts_embedding, tokenized_prompt)
            for module in self.text_feature_projection:
                text_features = module(features)

            # text_features = self.text_feature_projection(features)
            object_features = hidden_states[-1]

            ref_gt_instances = []
            for data_sample in batch_data_samples:
                data_sample.ref_gt_instances.bboxes = data_sample.ref_gt_instances.bboxes.to(final_layers_outputs_classes.device)
                data_sample.ref_gt_instances.labels = data_sample.ref_gt_instances.labels.to(final_layers_outputs_classes.device)
                ref_gt_instances.append(data_sample.ref_gt_instances)

        if self.with_cata_retrieval:
            cata_embedding, tokenized_cata = self.token_to_embedding(self.tokenized_class)
            cata_features = self.text_encoder(cata_embedding, tokenized_cata)
            for module in self.text_feature_projection:
                cata_features = module(cata_features)
            
            object_features = hidden_states[-1]

            cata_features_list = []
            cata_instances_list = []
            for data_sample in batch_data_samples:

                labels = data_sample.gt_instances.labels

                # 对当前 batch 的索引进行去重
                unique_labels, inverse_labels = torch.unique(labels, return_inverse=True)

                # 提取不重复的类特征
                unique_features = cata_features[unique_labels]

                # 初始化合并后的 bboxes
                merged_instances = []
                # 根据 inverse_indices 合并相同索引的 bbox
                for j in range(unique_labels.size(0)):
                    # 选择所有属于该类的 bbox，并计算平均值来合并
                    merged_instances.append(data_sample.gt_instances[inverse_labels == j])

                

                # 保存提取的特征和合并的 bbox
                cata_features_list.append(unique_features)
                cata_instances_list.append(merged_instances)

        tgt = []
        logits = []
        for i in range(b):
            bbox_pred = final_layers_outputs_coords[i]
            img_h, img_w = batch_data_samples[i].metainfo['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,img_h]).unsqueeze(0)

            query_features = []
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
            bbox_pred = bbox_pred * factor
            pred_instances = InstanceData(scores=final_layers_outputs_classes[i], bboxes=bbox_pred)
            if self.with_ref_retrieval:
                assign_result = self.assigner.assign(pred_instances, ref_gt_instances[i], img_meta=batch_data_samples[i].metainfo)
                query_features.append(text_features[i])
                tgt.append(assign_result.gt_inds)

            if self.with_cata_retrieval:
                for j in range(len(cata_instances_list[i])):
                    query_features.append(cata_features_list[i][j])
                    cata_gt_instances = cata_instances_list[i][j]
                    cata_assign_result = self.assigner.assign(pred_instances, cata_gt_instances, img_meta=batch_data_samples[i].metainfo)
                    cata_tgt = torch.clamp(cata_assign_result.gt_inds, max=1.0)
                    tgt.append(cata_tgt)

            query_features = torch.stack(query_features)
            logits.append(torch.matmul(query_features, object_features[i].transpose(0,1)).squeeze(1))

        tgt = torch.stack(tgt)
        logits = torch.concat(logits)

        losses = dict()
        retrieval_loss = self.loss_retrieval(logits, tgt)

        losses['retrieval_loss'] = retrieval_loss*b

        return losses
    
    def predict(self, hidden_states, final_layers_outputs_coords, batch_data_samples):
        if self.with_ref_retrieval:
            texts = []
            for d in batch_data_samples:
                texts.extend(d.text)
            tokenized_text = self.tokenizer(texts).cuda()
            prompts_embedding, tokenized_prompt = self.token_to_embedding(tokenized_text)

            features = self.text_encoder(prompts_embedding, tokenized_prompt)

            for module in self.text_feature_projection:
                text_features = module(features)

            object_features = hidden_states[-1]

            # similarity = torch.cosine_similarity(text_features.unsqueeze(1), object_features, dim=-1)
            similarity = torch.matmul(text_features.unsqueeze(1), object_features.transpose(1, 2)).squeeze(1)

            # similarity_softmax = torch.softmax(similarity, dim=-1)

            _, max_index = torch.max(similarity, dim=1)

            # pred_bbox = bbox_cxcywh_to_xyxy(final_layers_outputs_coords)


            best_bbox = final_layers_outputs_coords.bboxes[max_index]

            return best_bbox, text_features
        
        return None, None


    def freeze_clip(self):
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
