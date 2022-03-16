import torch
from torch import nn
import os
import torch.nn.functional as F
from audtorch.metrics import ConcordanceCC, PearsonR

class CCCLoss(nn.Module):
    '''
    目前lrc用的ccc loss版本
    '''
    def __init__(self, reduction='mean', batch_first=True, batch_compute=False):
        super(CCCLoss, self).__init__()
        self.reduction = reduction
        self.batch_first = batch_first
        self.batch_compute = batch_compute
        #self.metric = ConcordanceCC(reduction=reduction, batch_first=batch_first)

    def forward(self, inputs, target, mask=None):
        if mask is not None:
            if self.batch_compute:
                inputs, target = inputs[mask != 0], target[mask != 0]
            else:
                inputs, target = inputs * mask, target * mask

        if self.batch_compute:
            inputs, target = inputs.reshape(1, -1), target.reshape(1, -1)
        a_mean, b_mean = torch.mean(inputs, dim=1), torch.mean(target, dim=1)
        a_var = torch.mean(torch.square(inputs), dim=1)-torch.square(a_mean)
        b_var = torch.mean(torch.square(target), dim=1)-torch.square(b_mean)
        cor_ab = torch.mean((inputs - a_mean.unsqueeze(1))*(target - b_mean.unsqueeze(1)), dim=1)

        ccc = 2 * cor_ab / (a_var + b_var + torch.square(a_mean - b_mean) + 1e-9)
        batch_size, length = target.shape[0], target.shape[1]
        loss = torch.sum((1 - ccc) * torch.sum(mask, dim=1) / length)
        if self.reduction == 'mean':
            loss = loss / torch.sum(mask) * length

        #loss = 1 - torch.sum(ccc) / torch.sum(mask) * mask.shape[1]
        #loss = torch.sum(1-ccc) if self.reduction == 'sum' else torch.mean(1-ccc)
        return loss