from copy import deepcopy
import types
import numpy as np
import torch
from torch import nn
import open_clip
# from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel
from typing import Callable, List, Optional, Sequence, Tuple, Union

def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


def forward_with_prompt(self, text, prompt=None):
    cast_dtype = self.transformer.get_cast_dtype()
    seq_len = text.shape[1]

    x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
    attn_mask = self.attn_mask

    if prompt is not None:
        prompt_len = prompt.shape[1]
        x = torch.cat([prompt, x[:, :-prompt_len,:]], dim=1)

    x = x + self.positional_embedding[:seq_len].to(cast_dtype)
    x = self.transformer(x, attn_mask=attn_mask)

    # x.shape = [batch_size, n_ctx, transformer.width]

    x = self.ln_final(x)
    pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

    if self.text_projection is not None:
        if isinstance(self.text_projection, nn.Linear):
            pooled = self.text_projection(pooled)
        else:
            pooled = pooled @ self.text_projection

    if self.output_tokens:
        return pooled, tokens

    return pooled


class MoERouter(nn.Module):
    def __init__(self, cfg, input_dim):
        super().__init__()
        self.num_experts = cfg.MODEL.MOE.PROMPT_NUM
        self.hidden_dim = cfg.MODEL.MOE.HIDDEN_DIM
        self.drop_out = cfg.MODEL.MOE.DROP_OUT
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.hidden_dim, self.num_experts)
        )
    
    def forward(self, x):
        logits = self.ffn(x)  # (bs, num_experts)
        return torch.softmax(logits, dim=-1)  # 归一化为概率分布
    

class PromptLearner(nn.Module):
    def __init__(self, cfg, text_encoder):
        super().__init__()
        self.prompt_len = cfg.MODEL.MOE.PROMPT_LEN
        self.prompt_num = cfg.MODEL.MOE.PROMPT_NUM
        self.input_dim = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        self.hidden_dim = cfg.MODEL.MOE.HIDDEN_DIM
        self.output_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        # 获取 token_embedding 结构
        vocab_size, embed_dim = text_encoder.token_embedding.weight.shape

        # 新建一个相同的 nn.Embedding 层
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # 复制权重
        self.token_embedding.weight.data.copy_(text_encoder.token_embedding.weight.data)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.output_dim, 
            nhead=8,  
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        ctx_vectors = torch.empty(self.prompt_num, self.prompt_len, self.output_dim)
        nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)
        self.router = MoERouter(cfg, self.output_dim)


    def forward(self, text):
        ctx = self.ctx
        x = self.token_embedding(text)  # (bs, prompt_len, dim)

        text_feature = self.transformer(x)  # (bs, prompt_len, dim)
        text_feature = text_feature.mean(dim=1)  # (bs, dim)
        prompt_weights = self.router(text_feature)  # (bs, num_experts)
        final_prompt = torch.einsum('bn,nld->bld', prompt_weights, ctx)  # (bs, prompt_len, dim)

        return final_prompt


class OpenclipTextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.bert_name = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        pretrained = cfg.MODEL.LANGUAGE_BACKBONE.PRETRAINED

        model, _, _ = open_clip.create_model_and_transforms(self.bert_name, pretrained=pretrained, force_custom_text=True)
        
        self.parallel_det = cfg.MODEL.PARALLEL_DET
        self.use_prompt = False
        if self.use_prompt:
            self.model = model.text
            self.model.forward = types.MethodType(forward_with_prompt, self.model)
            self.prompt_learner = PromptLearner(cfg, self.model)
        else:
            self.model = model.text


        del model
        torch.cuda.empty_cache()

    def forward(self, x, task=None):
        input = x["input_ids"] # (bs, seq_len)
        mask = x["attention_mask"] # (bs, seq_len)

        if self.parallel_det and task == "detection":
            hidden_state = self.model(
                input,
                # attention_mask=mask_new,
                # output_hidden_states=True,
            )
        else:
            if self.use_prompt:
                prompt = self.prompt_learner(input)
            
                hidden_state = self.model(
                    input,
                    prompt
                    # attention_mask=mask,
                    # output_hidden_states=True,
                )
            else:
                hidden_state = self.model(
                    input,
                    # attention_mask=mask,
                    # output_hidden_states=True,
                )

        ret = {
            # "aggregate": aggregate,
            # "embedded": embedded,
            "masks": mask,
            "hidden": hidden_state,
        }
        return ret
    


