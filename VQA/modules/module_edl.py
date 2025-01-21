import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config

class EDL(nn.Module):
    def __init__(self, config: Config):
        super(EDL, self).__init__()
        self.batch_size = config.batch_size
        self.embed_dim = config.embed_dim
        self.num_frames = config.num_frames

    def loglikelihood_loss(self, target, alpha, device=None):
        target = target.to(device)
        alpha = alpha.to(device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((target - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood

    def mse_loss(self, target, alpha):

        loglikelihood = self.loglikelihood_loss(target, alpha)

        return loglikelihood

    def forward(self, vqa_logits, label, num_classes):

        label = torch.nn.functional.one_hot(label, num_classes=num_classes)

        '''Evidence Generation of Ablation Study'''
        evidence = F.relu(vqa_logits)
        # evidence = torch.exp(matrix)
        # evidence = F.softplus(matrix)
        alpha = evidence + 1

        loss = torch.mean(self.mse_loss(label, alpha))

        return loss