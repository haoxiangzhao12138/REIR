import logging
import math
import types
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import open_clip
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
from open_clip.timm_model import TimmModel
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
import timm
from timm.models import checkpoint_seq
from functools import partial
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ["SigLIP", "SimpleFeaturePyramid", "get_vit_lr_decay_rate"]

# def get_abs_pos(abs_pos, has_cls_token, hw):
#     """
#     Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
#         dimension for the original embeddings.
#     Args:
#         abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
#         has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
#         hw (Tuple): size of input image tokens.

#     Returns:
#         Absolute positional embeddings after processing with shape (1, H, W, C)
#     """
#     h, w = hw
#     if has_cls_token:
#         abs_pos = abs_pos[:, 1:]
#     xy_num = abs_pos.shape[1]
#     size = int(math.sqrt(xy_num))
#     assert size * size == xy_num

#     if size != h or size != w:
#         new_abs_pos = F.interpolate(
#             abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
#             size=(h, w),
#             mode="bicubic",
#             align_corners=False,
#         )

#         return new_abs_pos
#     else:
#         return abs_pos.reshape(1, h, w, -1)

# def new_forward(self, x: torch.Tensor):
#     x = self.conv1(x)  # shape = [*, dim, h, w]
#     b,c,h,w = x.shape
#     if self.positional_embedding is not None:
#         x = x + get_abs_pos(self.positional_embedding.unsqueeze(0), True, (x.shape[2], x.shape[3]))
#     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, dim, hw]
#     x = x.permute(0, 2, 1)  # shape = [*, hw, dim]

#     # class embeddings and positional embeddings
#     x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
#     # shape = [*, grid ** 2 + 1, width]


#     x = self.patch_dropout(x)
#     x = self.ln_pre(x)
#     x = self.transformer(x)

#     if self.attn_pool is not None:
#         if self.attn_pool_contrastive is not None:
#             # This is untested, WIP pooling that should match paper
#             x = self.ln_post(x)  # TBD LN first or separate one after each pool?
#             tokens = self.attn_pool(x)
#             if self.attn_pool_type == 'parallel':
#                 pooled = self.attn_pool_contrastive(x)
#             else:
#                 assert self.attn_pool_type == 'cascade'
#                 pooled = self.attn_pool_contrastive(tokens)
#         else:
#             # this is the original OpenCLIP CoCa setup, does not match paper
#             x = self.attn_pool(x)
#             x = self.ln_post(x)
#             pooled, tokens = self._global_pool(x)
#     elif self.final_ln_after_pool:
#         pooled, tokens = self._global_pool(x)
#         pooled = self.ln_post(pooled)
#     else:
#         x = self.ln_post(x)
#         pooled, tokens = self._global_pool(x)

#     # if self.proj is not None:
#     #     pooled = pooled @ self.proj
#     tokens = tokens.view(b,h,w,c)
#     return tokens

# def _expand_token(token, batch_size: int):
#     return token.view(1, 1, -1).expand(batch_size, -1, -1)
def forward_features(self, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_embed(x)
    b,h,w,c = x.shape
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    x = self.blocks(x)

    x = self.norm(x)

    return x.view(b,h,w,c).contiguous()


def forward(self, x: torch.Tensor, use_pool_head=False) -> torch.Tensor:
    x = self.patch_embed(x)
    b,h,w,c = x.shape
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    x = self.blocks(x)

    x = self.norm(x)

    feature_map = x.view(b,h,w,c).contiguous()
    img_feature = None
    if use_pool_head:
        img_feature = self.forward_head(x)
    return img_feature, feature_map 


class SigLIP(Backbone):
    def __init__(
        self,
        siglip_model,
        patch_size=16,
        embed_dim=768,
        out_feature="last_feat",
        cfg=None,
    ):
        super().__init__()
        self.use_clip_pool_head = cfg.MODEL.USE_CLIP_POOL_HEAD
        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]


        # In our method, we don't use backbone feature with stride 4
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Identity() 
        self.fpn3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.apply(self._init_weights)

        timm_kwargs = {}
        timm_kwargs['dynamic_img_size'] = True
        self.vision_model = timm.create_model(
            siglip_model,
            num_classes=0,
            global_pool='map',
            pretrained=True,
            pretrained_cfg_overlay=dict(file=cfg.MODEL.LANGUAGE_BACKBONE.PRETRAINED),
            **timm_kwargs,
        )
        self.vision_model.forward = types.MethodType(forward, self.vision_model)
        pass

        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        img_feature, feature_map = self.vision_model(x, self.use_clip_pool_head)

        xp = feature_map.permute(0, 3, 1, 2).contiguous() # (b, h, w, c) --> (b, c, h, w)
        
        features = []
        ops = [self.fpn1, self.fpn2, self.fpn3]
        for i in range(len(ops)):
            features.append(ops[i](xp))
        rets = {"res{}".format(u + 3): v for (u,v) in enumerate(features)}

        return rets, img_feature
    
# ('ViT-B-16-SigLIP', 'webli'), 
# ('ViT-B-16-SigLIP-256', 'webli'), 
# ('ViT-B-16-SigLIP-i18n-256', 'webli'), 
# ('ViT-B-16-SigLIP-384', 'webli'), 
# ('ViT-B-16-SigLIP-512', 'webli'), 
# ('ViT-L-16-SigLIP-256', 'webli'), 
# ('ViT-L-16-SigLIP-384', 'webli'), 
# ('ViT-SO400M-14-SigLIP', 'webli'), 
# ('ViT-SO400M-16-SigLIP-i18n-256', 'webli'), 
# ('ViT-SO400M-14-SigLIP-378', 'webli'), 
# ('ViT-SO400M-14-SigLIP-384', 'webli')
@BACKBONE_REGISTRY.register()
class D2SigLIP(SigLIP, Backbone):
    def __init__(self, cfg, input_shape):
        if cfg.MODEL.VIT.MODEL_TYPE == "vit_base_patch16_siglip_384":
            embed_dim = 768
        elif cfg.MODEL.VIT.MODEL_TYPE == "vit_base_patch16_siglip_512":
            embed_dim = 768
        elif cfg.MODEL.VIT.MODEL_TYPE == "vit_large_patch16_siglip_384":
            embed_dim = 1024
        
        
        super().__init__(
            siglip_model=cfg.MODEL.VIT.MODEL_TYPE,
            patch_size=16,
            embed_dim=embed_dim,
            out_feature="last_feat",
            cfg=cfg,
        )

        self._out_features = cfg.MODEL.VIT.OUT_FEATURES

        self._out_feature_strides = {
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res3": embed_dim // 2,
            "res4": embed_dim,
            "res5": embed_dim,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y, image_features = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs, image_features

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

