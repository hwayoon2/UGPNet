import sys
sys.path.append('..')
from archs.blocks import ResBlock
import torch
import torch.nn as nn
from utils import initialize_weights

class FusionNet(nn.Module):
    def __init__(self, num_block=8, chan=64):
        super(FusionNet, self).__init__()
        self.num_block = num_block
        self.conv = nn.Conv2d(chan,64,3,1,1)
        self.body = nn.Sequential(
            *[ResBlock() for _ in range(num_block)],
        )
        self.last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        initialize_weights(self)


    def forward(self, F_g, F_c):
        
        x = self.conv(F_g) + F_c

        x = self.body(x)
        x = self.last(x)
        return x
