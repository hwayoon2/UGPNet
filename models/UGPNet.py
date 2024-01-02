import sys
sys.path.append('..')
import torch
import torch.nn as nn
from image_tools import ganimage_preprocess
import torch.nn.functional as F
from models.resmodule import ResModule
from models.synmodule import SynModule
from models.fusmodule import FusionNet
from archs.RRDBNet import RRDBNetX8
from archs.nafnet import NAFNet, NAFNetLocal
from archs.hinet_arch import HINet

class UGPNet(nn.Module):
    def __init__(self, args):
        super(UGPNet, self).__init__()
        
        self.args = args
        
        if "rrdb" in args.type:
            self.resmodule = RRDBNetX8(3, 3)
        else:
            self.resmodule = ResModule(args)
            
        self.synmodule = SynModule(args)

        chan = 128 if args.church else 64
        self.fusmodule = FusionNet(chan=chan)

    def load_ckpt(self):
        if "rrdb" in self.args.type:
            self.resmodule.load_state_dict(torch.load(self.args.resmodule_path)['params_ema'])

        else:
            self.resmodule.load_state_dict(torch.load(self.args.resmodule_path))    
        self.synmodule.load()
        self.fusmodule.load_state_dict(torch.load(self.args.fusmodule_path))

    def forward(self, lq_img, return_all=False):
        
        I_R, F_R = self.resmodule(lq_img, return_f=True)
        I_G, F_G = self.synmodule(I_R, return_f=True)
        I_F = self.fusmodule(F_G, F_R)

        if return_all:
            return I_R, I_G, I_F

        return I_F