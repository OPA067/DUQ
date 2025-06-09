import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module_local import MultiHeadSelfAttention

class prob_embed(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super(prob_embed, self).__init__()

        self.embed_dim = d_in

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.fc1 = nn.Linear(d_in, d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, global_feat, local_feat, out, pad_mask=None):

        residual, attn = self.attention(local_feat, pad_mask)

        fc_out = self.fc1(global_feat)
        global_feat = self.fc(residual) + fc_out

        mu = l2_normalize(out)
        sigma = global_feat
        return {
            'mu': mu,
            'sigma': sigma,
        }

def sample_gaussian_tensors(mu, sigma, num_samples):

    eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)
    samples = eps.mul(torch.exp(sigma.unsqueeze(1))).add_(mu.unsqueeze(1))
    return samples

def l2_normalize(tensor, axis=-1):

    return F.normalize(tensor, p=2, dim=axis)