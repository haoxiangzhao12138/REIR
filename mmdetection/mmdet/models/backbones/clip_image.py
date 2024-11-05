import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmengine.model import BaseModule
import open_clip

@MODELS.register_module()
class ClipImageEncoder(BaseModule):
    def __init__(self, 
                 name='ViT-H-14', 
                 pretrained='/public/haoxiangzhao/weights/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin',
                 freeze_encoder=False):
        super().__init__()
        self.clip, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(name, pretrained=pretrained, force_custom_text=True,)
        self.image_encoder = self.clip.visual
        del self.clip
        if freeze_encoder:
            self.freeze_clip()


    def forward(self, x):
        # text = self.tokenizer(x)
        device = self.text_encoder.attn_mask.device

        # print(x)
        text = x.to(device=device)
        features = self.text_encoder(text)

        return features

    def freeze_clip(self):
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
