import torch.nn as nn
import torch.nn.functional as F

from models.module_agg import MultiHeadSelfAttention

class prob_embed(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super(prob_embed, self).__init__()
        self.num_heads = num_heads
        self.attention = MultiHeadSelfAttention(self.num_heads, embed_dim)
        self.feat_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim), )
        self.ln = nn.LayerNorm(embed_dim)

    def l2_normalize(self, tensor, axis=-1):
        return F.normalize(tensor, p=2, dim=axis)

    def forward(self, feat, out):

        feat = self.feat_mlp(self.attention(feat))
        sigma = self.ln(feat)
        mu = self.l2_normalize(out)
        return {
            'mu': mu,
            'sigma': sigma,
        }