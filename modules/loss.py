import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from config.base_config import Config

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale):

        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0

class KLdivergence(nn.Module):
    def __init__(self):
        super(KLdivergence, self).__init__()

    def kl_divergence(self, mu, logsigma):
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum()

    def forward(self, sampled_video, video_sigma, sampled_text, text_sigma):
        vib_loss = self.kl_divergence(sampled_video.mean(dim=1), video_sigma) + self.kl_divergence(sampled_text.mean(dim=1), text_sigma)
        return vib_loss

class LossFactory:
    @staticmethod
    def get_loss(config_loss):
        if config_loss == 'clip':
            return CLIPLoss()
        else:
            raise NotImplemented
