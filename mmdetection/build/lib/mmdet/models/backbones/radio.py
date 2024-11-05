
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmengine.model import BaseModule
from einops import rearrange
import copy
import math
from typing import Optional
from mmengine.config import ConfigDict
#
import torch
import os
import torch.nn.functional as F
from torch import Tensor, nn
from mmengine.device import get_device
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class Radio(BaseModule):
    def __init__(
            self,
            path='/home/haoxiangzhao/.cache/torch/hub/NVlabs_RADIO_main',
            name='radio_model',
            weights='/storage-root/datasets/haoxiangzhao/radio/radio_v2.1_bf16.pth.tar',
            version='radio_v2.1',
            progress=True,
            skip_validation=True,
            source='local',
            with_MHCA=True,
            focus_layers_num=1,
            pre_norm: bool = False,
            hidden_dim: int = 512,
            nheads: int = 8,
            dropout=0.1,
            dim_feedforward=2048,
            text_encoder=ConfigDict,
            text_focus_prompt=list,
            init_cfg: OptMultiConfig = None
    ):
        super().__init__(init_cfg=init_cfg)
        self.with_MHCA = with_MHCA
        self.radio = torch.hub.load(path, name, checkpoint_path=weights, version=version, progress=progress,
                                    skip_validation=skip_validation, source=source, )
        if with_MHCA:   
            self.MHCA = nn.ModuleList()
            self.FFN = nn.ModuleList()
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

                self.FFN.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )
            self.text_encoder = MODELS.build(text_encoder)
            self.register_buffer('text_prompt_features', self.text_encoder.text_encoder(self.text_encoder.tokenizer(text_focus_prompt)))
            del self.text_encoder

    def forward(self, x):
        # print(x)
        b, _, h, w = x.shape[:]
        rows = h // 16
        cols = w // 16

        summary, features = self.radio(x)

        if self.with_MHCA:
            image_features = features.permute(1,0,2)
            text_features = self.text_prompt_features.repeat(b,1,1).permute(1,0,2).detach()

            # torch.autograd.set_detect_anomaly(True)

            for i in range(self.focus_layers_num):
                image_features, avg_attn = self.MHCA[i](
                    image_features,
                    text_features
                )

                image_features = self.FFN[i](
                    image_features
                )

            final_features = features + image_features.permute(1,0,2)
        else:
            final_features = features

            
        features = rearrange(final_features, 'b (h w) c -> b c h w', h=rows, w=cols)

        return features
        # return tuple(features)


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
