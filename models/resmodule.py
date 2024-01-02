import sys
sys.path.append('..')
from archs.nafnet import NAFNet, NAFNetLocal
from archs.hinet_arch import HINet
from archs.blocks import ConvBlock, ToImage
import torch.nn as nn
import torch
import torch.nn.functional as F
from image_tools import ganimage_preprocess
from utils import initialize_weights

class ResidualEncoder(nn.Module):
    def __init__(self, chan=32):
        super(ResidualEncoder,self).__init__()
        self.input_encoder = ConvBlock(3, chan)
        self.merge_decoder = nn.ModuleList()
        self.merge_decoder.append(ConvBlock(chan, 64))
        self.merge_decoder.append(ConvBlock(64, 64))
        self.to_img = ToImage(64)
        initialize_weights(self)

    def forward(self, I_input, F_regression):
        F_input = self.input_encoder(I_input)
        f = F_regression + F_input
        for i in range(2):
            f = self.merge_decoder[i](f)
        I_output = self.to_img(f)
        return I_output, f

class ResModule(nn.Module):
    def __init__(self, args):

        super(ResModule, self).__init__()

        self.args = args
        if "nafnet" in args.type:
            chan = 32
            if "deblur" in args.type:
                self.regression = NAFNetLocal(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
            else: #denoise
                self.regression = NAFNet(img_channel=3, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
        elif "hinet" in args.type:
            if "deblur" in args.type:
                chan = 64
                self.regression = HINet(wf=64, hin_position_left=3, hin_position_right=4)
            else:
                chan = 32
                self.regression = HINet(wf=32)
            
        self.encoder = ResidualEncoder(chan=chan)
    
    def load_reg(self):
        self.regression.load_state_dict(torch.load(self.args.regression_path)['params'])

    def forward(self, I_input, return_f=False):
        F_regression = self.regression(I_input, return_f=True)
        I_output, F_res = self.encoder(I_input, F_regression)
        if return_f:
            return I_output, F_res
        return I_output