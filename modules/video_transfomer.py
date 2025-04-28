import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config

class video_transformer(nn.Module):
    def __init__(self, config: Config):
        super(video_transformer, self).__init__()
        self.embed_dim = config.embed_dim
        dropout = config.transformer_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.MLP = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, 1),
        )

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, video_embeds):
        """ text_embeds: [batch_size, embed_dim]
            video_embeds: [batch_size, frame_len, embed_dim]
            Here we add other video frame level feature pooling methods."""
        """ 1) Mean: 
            video_embeds = torch.mean(video_embeds, dim=1)
            return video_embeds"""
        """ 2) MLP:
            video_weights = self.MLP(video_embeds).squeeze(-1)
            video_embeds = torch.einsum("afd,af->ad", [video_embeds, video_weights])
            return video_embeds"""
        """ 3) Self-Attention:"""

        """ 4) Cross-Attention:"""
        # nn.LayerNorm
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # nn.Linear, Q = t \in [B, D] and K,V = v \in [B, F, D]
        q = self.q_proj(text_embeds)
        k = self.k_proj(video_embeds)
        v = self.v_proj(video_embeds)

        q = q.unsqueeze(1)
        k = k.transpose(1, 2)
        attention_logits = torch.matmul(q, k)
        attention_logits = attention_logits / math.sqrt(self.embed_dim)

        # Attention weights
        attention_weights = F.softmax(attention_logits, dim=2)

        attention = torch.matmul(attention_weights, v)
        attention = attention.squeeze(1)

        # Attention
        attention = self.out_proj(attention)

        attn_out = self.layer_norm2(attention)
        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        # video features output \in [B, D]
        return out
