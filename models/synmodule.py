import sys
sys.path.append('..')
import torch
import torch.nn as nn
import numpy as np
from archs.blocks import ConvBlock
from models.stylegan2 import StyleGAN2GeneratorF, StyleGAN2GeneratorFChurch
from image_tools import ganimage_preprocess
from torch.nn import functional as F
from utils import initialize_weights

class Fencoder(nn.Module):
    def __init__(self, spatial_size=16, encoder_input_size=512, init_weights=True):
        super(Fencoder, self).__init__()

        in_channels = 3
        encoder_output_size = spatial_size

        self.block_nums = int(np.log2(encoder_input_size//encoder_output_size))+1

        channels = {
            8: int(512),
            16: int(512),
            32: int(256),
            64: int(128),
            128: int(64),
            256: int(32),
            512: int(16),
        }

        self.downblocks = nn.ModuleList()

        for i in range(self.block_nums):
            out_channels = channels[encoder_input_size//(2**i)]
            if i == self.block_nums-1:
                self.downblocks.append(ConvBlock(in_channels, out_channels, down=False, batch_norm=True))
            else:
                self.downblocks.append(ConvBlock(in_channels, out_channels, down=True, batch_norm=True))
            in_channels = out_channels

        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1) #1x1 conv

        initialize_weights(self)

    def forward(self, x):

        for i in range(self.block_nums):
            x= self.downblocks[i](x)

            if x.shape[2]==32:
                w_input = x #32

        x = self.conv1x1(x)

        return x, w_input

class Wmodule(nn.Module):
    def __init__(self, in_c=128, out_c = 512, init_weights=True, wpdim=18, spatial_size=32):
        super(Wmodule, self).__init__()
        self.style_count = wpdim - 4
        self.out_c = out_c
        self.spatial = spatial_size
        num_pools = int(np.log2(spatial_size))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = nn.ModuleList()
        for idx in range(self.style_count):
            self.linear.append(nn.Linear(out_c, out_c))

        initialize_weights(self)

    def forward(self, x):
        
        latents = []
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        w1=self.linear[0](x)

        #dummy w (dont use)
        for idx in range(5):
            latents.append(w1)

        for idx in range(1, self.style_count):
            latents.append(self.linear[idx](x))
        w = torch.stack(latents, dim=1)

        return w


class SynModule(nn.Module):
    def __init__(self, args):
        super(SynModule, self).__init__()
        self.args = args
        self.generator = StyleGAN2GeneratorF() if not args.church else StyleGAN2GeneratorFChurch(basecode_size=args.spatial_size)
        self.fencoder = Fencoder(spatial_size=args.spatial_size) if not args.church else Fencoder(spatial_size=args.spatial_size, encoder_input_size=256)
        wpdim = 16 if not args.church else 14
        self.wmodule = Wmodule(wpdim=wpdim)
    
    def load(self):
        ckpt = torch.load(self.args.synmodule_path)
        self.generator.load_state_dict(ckpt['generator'])
        self.fencoder.load_state_dict(ckpt['fencoder'])
        self.wmodule.load_state_dict(ckpt['wmodule'])
    
    def load_generator_only(self):
        self.generator.load_state_dict(torch.load(self.args.generator_path)['params_ema'])
        print("load generator only")

    def forward(self, encoder_input, return_f=False):
        basecode, w_input = self.fencoder(encoder_input)
        detailcode = self.wmodule(w_input)
        x_rec, g_feature = self.generator(detailcode,input_is_basecode=True,basecode=basecode,randomize_noise=self.args.randomize_noise, return_f=True)
        x_rec = ganimage_preprocess(x_rec) #0-1
        
        if return_f:
            return x_rec, g_feature
        else:
            return x_rec
