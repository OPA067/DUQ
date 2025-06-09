import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_mha_heads=1):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, s_feat, f_feat):

        num_texts, _ = s_feat.shape
        q = self.q_proj(s_feat)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        q = q.permute(1, 2, 0)

        num_vids, num_frames, _ = f_feat.shape
        k = self.k_proj(f_feat)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)

        v = self.v_proj(f_feat)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 3, 1)

        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        attention = v @ attention_weights
        attention = attention.permute(0, 3, 1, 2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        o = self.out_proj(attention)
        return o

class frame_transformer(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(frame_transformer, self).__init__()
        self.embed_dim = embed_dim

        self.cross_attn = MultiHeadedAttention(embed_dim=self.embed_dim, num_mha_heads=1)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, s_feat, f_feat):

        s_feat = self.layer_norm1(s_feat)
        f_feat = self.layer_norm1(f_feat)

        attn_out = self.cross_attn(s_feat, f_feat)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out

