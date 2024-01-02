import torch
import torch.nn as nn
import torch.nn.functional as F

class ToImage(nn.Module):
    '''
    make feature to 3 channel image
    input : feature with in_channel
    output : 3 channel image
    '''
    def __init__(self, in_channel):
        super(ToImage, self).__init__()
        self.body = nn.Sequential(
                nn.Conv2d(in_channel, 3, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    def forward(self, x):
        return self.body(x)

class ConvBlock(nn.Module):
    '''
    Convolution Block
    [Conv - (BatchNormalization) - leakyReLU] X 3
    '''
    def __init__(self, in_channels, out_channels, down=False, downscale=2, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.ModuleList()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.ModuleList()
        self.down = down        
        self.non_linear = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for _ in range(3):
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                self.bn.append(nn.BatchNorm2d(out_channels))
            in_channels=out_channels
        
        if self.down:
            self.avgpool = nn.AvgPool2d(kernel_size=downscale, stride=downscale)

    def forward(self, x):
        for i in range(3):
            x = self.conv[i](x)

            if self.batch_norm:
                x = self.bn[i](x)
            
            x = self.non_linear(x)

        if self.down:
            x = self.avgpool(x)
        return x

class ResBlock(nn.Module):
    """
    Basic residual block from SRNTT.
    https://github.com/S-aiueo32/srntt-pytorch/blob/4ea0aa22a54a2d1b1f19c4a43596a693b9e7c067/models/srntt.py
    Parameters
    ---
    n_filters : int, optional
        a number of filters.
    """

    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):

        out = self.body(x) + x
        out = self.relu(out)

        return out