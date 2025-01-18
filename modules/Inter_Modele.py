import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config

class Inter_Modele(nn.Module):
    def __init__(self, config: Config):
        super(Inter_Modele, self).__init__()
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

    def forward(self, matrix):

        target = 1 - torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        '''
        [0 1 1 1
         1 0 1 1
         1 1 0 1
         1 1 1 0]
        '''
        evidence = F.relu(matrix)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        loss = (torch.mean(self.mse_loss(target, alpha)) + torch.mean(self.mse_loss(target, alpha.T))) / 2.0
        return loss, matrix.shape[0] / S

