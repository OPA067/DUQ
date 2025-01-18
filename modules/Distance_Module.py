import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config

class Distance_Module(nn.Module):
    def __init__(self, config: Config):
        super(Distance_Module, self).__init__()
        self.batch_size = config.batch_size
        self.embed_dim = config.embed_dim
        self.num_frames = config.num_frames


    def forward(self, Prob_text, Prob_video):

        b = Prob_text.size(0)

        dis_matrix = torch.zeros(b, b, device=Prob_text.device)

        for i in range(b):
            for j in range(b):
                t = Prob_text[i] / Prob_text[i].norm(dim=-1, keepdim=True)
                v = Prob_video[i] / Prob_video[i].norm(dim=-1, keepdim=True)
                distances = 1 - torch.mm(t, v.t())
                if i == j:
                    dis_matrix[i, j] = distances.min()
                else:
                    dis_matrix[i, j] = distances.max()

        # 0-1
        min_val = torch.min(dis_matrix)
        max_val = torch.max(dis_matrix)
        dis_matrix = (dis_matrix - min_val) / (max_val - min_val)

        return dis_matrix
