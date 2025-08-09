import numpy as np
import torch
from torch import nn
from collections import OrderedDict

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super(ResidualAttentionBlock, self).__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))  # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, num_heads, embed_dim):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ResBlock = ResidualAttentionBlock(d_model=self.embed_dim, n_head=self.num_heads)
        self.feat_w = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, 1),
        )

    def forward(self, x, mask=None):
        x = x + self.ResBlock(x)
        attn = self.feat_w(x).squeeze(-1)
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask.bool(), -np.inf)
        attn = torch.softmax(attn, dim=-1)

        output = torch.einsum("bnd,bn->bd", [x, attn])
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output

class module_agg(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super(module_agg, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attention = MultiHeadSelfAttention(self.num_heads, self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, feat):
        feat = self.mlp(self.attention(feat))
        feat = self.ln(feat)
        return feat