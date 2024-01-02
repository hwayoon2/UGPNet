import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import lpips


class LpipsLoss(nn.Module):

    def __init__(self):
        super(LpipsLoss, self).__init__()
        self.lpips_fn=lpips.LPIPS(net='vgg').cuda()
        self.lpips_fn.net.requires_grad_(False)


    def forward(self, gt, output):
        output = output*2.-1
        gt = gt*2.-1
        lpips_output = torch.nn.functional.interpolate(output, size=(256,256), mode='bicubic')
        lpips_gt = torch.nn.functional.interpolate(gt, size=(256,256), mode='bicubic')
        lpips_loss = torch.mean(self.lpips_fn(lpips_output, lpips_gt))
        return lpips_loss