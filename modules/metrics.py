import numpy as np
import torch
import torch.nn.functional as F
import scipy.stats
from config.all_config import gen_log
import gc

def np_softmax(X, theta=1.0, axis=None):
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1:
        p = p.flatten()
    return p

def sim_matrix_training(text_embeds, vid_embeds_pooled):

    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

    sims = torch.mm(text_embeds, vid_embeds_pooled.t())

    return sims

def metrics(x):
    # x = x[:, 0, :]
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['Rsum'] = metrics['R1'] + metrics['R5'] + metrics['R10']
    metrics['R50'] = float(np.sum(ind < 50)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MdR"] = metrics['MR']
    metrics["MnR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def pad_and_stack_dict_to_tensor(input, order, d=512):
    max_length = max([input[k].shape[0] for k in input])

    padded_input = {
        k: torch.cat([input[k], torch.full((max_length - input[k].shape[0], d), float("-inf"), device=input[k].device)])
        for k in input}

    padded_stacked_input = torch.stack([padded_input[k] for k in order], dim=0)
    return padded_stacked_input
