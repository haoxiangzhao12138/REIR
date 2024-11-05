
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from einops import rearrange
import copy
import math
from typing import Optional
from mmengine.config import ConfigDict
import torch.distributed as dist
import torch
import os
import warnings

import torch.nn.functional as F
from torch import Tensor, nn
from mmengine.device import get_device
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from transformers import AutoModel
from timm.models.vision_transformer import VisionTransformer

@MODELS.register_module()
class Radio(BaseModule):
    def __init__(
            self,
            with_MHCA=True,
            focus_layers_num=1,
            pre_norm: bool = False,
            hidden_dim: int = 512,
            nheads: int = 8,
            dropout=0.1,
            text_focus_prompt=list,
            path='/storage-root/datasets/haoxiangzhao/radio/NVlabs_RADIO_main',
            name='radio_model',
            weights='/storage-root/datasets/haoxiangzhao/radio/radio_v2.1_bf16.pth.tar',
            version='radio_v2.1',
            progress=True,
            skip_validation=True,
            adaptor_names='clip',
            source='local',
            freeze_backbone=True,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.freeze_backbone = freeze_backbone
        # Instantiate a RADIO model.
        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     # Pull the model on rank 0 first.
        #     model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, token=token)
        # if dist.is_initialized():
        #     dist.barrier()
        #     if dist.get_rank() > 0:
        #         # Now pull the model from cache on other ranks.
        #         model = AutoModel.from_pretrained(repo_id, trust_remote_code=True, token=token)
        model = torch.hub.load(path, name, version=weights, progress=progress,
                                    skip_validation=skip_validation, source=source, adaptor_names=adaptor_names,)

        self.base_model = model
        

        self.base_model.input_conditioner = nn.Identity()
        # self.input_conditioner = nn.Identity()

        self.with_MHCA = with_MHCA
        # self.radio = torch.hub.load(path, name, checkpoint_path=weights, version=version, progress=progress,
        #                             skip_validation=skip_validation, source=source, )
        if with_MHCA:   
            self.MHCA = nn.ModuleList()
            self.focus_layers_num = focus_layers_num
            for _ in range(self.focus_layers_num):
                # self.text_focus.append(
                #     SelfAttentionLayer(
                #         d_model=hidden_dim,
                #         nhead=nheads,
                #         dropout=0.0,
                #         normalize_before=pre_norm,
                #     )
                # )
                self.MHCA.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    )
                )

                # self.FFN.append(
                #     FFNLayer(
                #         d_model=hidden_dim,
                #         dim_feedforward=dim_feedforward,
                #         dropout=0.0,
                #         normalize_before=pre_norm,
                #     )
                # )
            self.text_encoder = self.base_model.adaptors['clip']
            self.register_buffer('text_prompt_features', self.text_encoder.encode_text(self.text_encoder.tokenizer(text_focus_prompt)))
            self.projection = nn.Linear(1024, 1280, bias=True)
            # self.text_prompt_features
            # del self.text_encoder
        
        if self.freeze_backbone:
            self.freeze_radio()

    def forward(self, x):
        # print(x)
        # for idx, (name, param) in enumerate(self.named_parameters()):
        #     print(f"Index: {idx}, Parameter Name: {name}, Requires Grad: {param.requires_grad}")
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(name)
        B, _, H, W = x.shape

        # Scale inputs to the range [0, 1].
        # x = x / 255.0
        output = self.base_model(x)
        if not self.with_MHCA:
            _, features = output['backbone']

            if isinstance(self.base_model.model, VisionTransformer):
                # Reshape
                B, _, C = features.shape

                if hasattr(self.base_model.model, "patch_generator"):
                    # Cropped Positional Embedding (CPE) case.
                    patch_height = patch_width = self.base_model.model.patch_generator.patch_size
                else:
                    # Standard ViT case.
                    patch_height, patch_width = self.base_model.model.patch_embed.patch_size
                final_features = features.reshape(B, math.ceil(H/patch_height), math.ceil(W/patch_width),  C).permute(0, 3, 1, 2).contiguous()
            return final_features
        else:
            _, features = output['backbone']
            image_features = features.permute(1,0,2)
            text_features = self.text_prompt_features.detach()
            text_features = text_features.repeat(B,1,1).permute(1,0,2)
            text_features = self.projection(text_features)

            # torch.autograd.set_detect_anomaly(True)

            for i in range(self.focus_layers_num):
                image_features, avg_attn = self.MHCA[i](
                    image_features,
                    text_features
                )

            final_features = features + image_features.permute(1,0,2)
            if isinstance(self.base_model.model, VisionTransformer):
                # Reshape
                B, _, C = features.shape

                if hasattr(self.base_model.model, "patch_generator"):
                    # Cropped Positional Embedding (CPE) case.
                    patch_height = patch_width = self.base_model.model.patch_generator.patch_size
                else:
                    # Standard ViT case.
                    patch_height, patch_width = self.base_model.model.patch_embed.patch_size
                final_features = final_features.reshape(B, math.ceil(H/patch_height), math.ceil(W/patch_width),  C).permute(0, 3, 1, 2).contiguous()
            return final_features
        # return tuple(features)
    
    def train(self, mode=True):
        """Intercept call."""
        # Drop a warning if mode is True.
        if mode:
            warnings.warn("RADIO is always in eval mode.")
        pass

    def init_weights(self):
        # This is a no-op as the model weights are loaded during instantiation.
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            self.base_model.eval()
            pass
        else:
            raise ValueError(f"Unhandled case: {self.init_cfg}")
    
    def freeze_radio(self):
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False


class CrossAttentionLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dropout=0.0,
                 activation='relu',
                 normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     tgt,
                     memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2, avg_attn = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt, avg_attn

    def forward_pre(self,
                    tgt,
                    memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2, avg_attn = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)

        return tgt, avg_attn

    def forward(self,
                tgt,
                memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class FFNLayer(nn.Module):

    def __init__(self,
                 d_model,
                 dim_feedforward=2048,
                 dropout=0.0,
                 activation='relu',
                 normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)
