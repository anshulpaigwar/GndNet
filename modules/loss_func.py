# owner: Anshul Paigwar
# Forked from https://github.com/ClementPinard/SfmLearner-Pytorch/blob/e1b5b0de40fe212f7ba8807e3037fedd0fe4f12f/loss_functions.py#L71

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# import ipdb as pdb




class SpatialSmoothLoss(torch.nn.Module):
    def __init__(self):
        super(SpatialSmoothLoss, self).__init__()

    def forward(self, pred_map):
        def gradient(pred):
            D_dy = pred[:, 1:] - pred[:, :-1]
            D_dx = pred[:, :, 1:] - pred[:, :, :-1]
            return D_dx, D_dy
        # pdb.set_trace()

        dx, dy = gradient(pred_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss = dx2.abs().mean(axis = (1,2)) + dxdy.abs().mean(axis = (1,2)) + dydx.abs().mean(axis = (1,2)) + dy2.abs().mean(axis = (1,2))
        return loss.mean()





class MaskedHuberLoss(torch.nn.Module):
    def __init__(self):
        super(MaskedHuberLoss, self).__init__()

    def forward(self, output, labels, mask):
        lossHuber = nn.SmoothL1Loss(reduction = "none").cuda()
        l = lossHuber(output*mask, labels*mask) #(B,100,100)
        l = l.sum(dim = (1,2))
        mask = mask.sum(dim = (1,2))
        l = l/mask
        return l.mean()
