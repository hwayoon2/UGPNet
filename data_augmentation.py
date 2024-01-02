import torch
from torch._C import device
import torchvision
import torchvision.transforms as transforms
import random as rand
import torchvision.transforms.functional as TF

class RandomAugmentation(object):
    def __init__(self, scale_range=[7/8, 9/8], translation_xrange=512, translation_yrange=100, rotation_range=20):
        self.scale = RandomScale(scale_range)
        self.translate = RandomTranslate(translation_xrange, translation_yrange)
        self.rotate = RandomRotate(rotation_range)

    def __call__(self, tensor, idx, opt=None):

        if idx%3==0:
            return self.scale(tensor, opt)
        elif idx%3==1:
            return self.translate(tensor, opt)
        else:
            return self.rotate(tensor, opt)

class RandomScale(object):
    def __init__(self, scale_range=[7/8, 9/8]):
        self.scale_range = scale_range

    def __call__(self, tensor, scale_factor=None):
        
        if scale_factor is None:
            scale_factor = rand.uniform(self.scale_range[0], self.scale_range[1])
        else:
            scale_factor = scale_factor[0]
        tensor = TF.affine(img = tensor, shear = 0, angle=0, translate=[0, 0], scale=scale_factor, fill=0)
        tensor = torch.clamp(tensor,min=0.0,max=1.0)

        return tensor, (scale_factor, None)

class RandomTranslate(object):
    def __init__(self, translation_xrange=512, translation_yrange=100):
        self.translation_xrange = translation_xrange
        self.translation_yrange = translation_yrange

    def __call__(self, tensor,trans_xy=None):
        if trans_xy is None:
            trans_x = rand.randint(-self.translation_xrange,self.translation_xrange+1)
            trans_y = rand.randint(-self.translation_yrange,self.translation_yrange+1)
        else:
            (trans_x, trans_y) = trans_xy
        tensor = TF.affine(img = tensor, shear = 0, angle=0, translate=[trans_x, trans_y], scale=1, fill=0)
        tensor = torch.clamp(tensor,min=0.0,max=1.0)

        return tensor, (trans_x, trans_y)

class RandomRotate(object):
    def __init__(self, rotation_range=20):
        self.rotation_range = rotation_range

    def __call__(self, tensor, angle=None):
        if angle is None:
            angle = rand.randint(-self.rotation_range,self.rotation_range)
        else:
            angle = angle[0]
        tensor = TF.affine(img = tensor, shear = 0, angle=angle, translate=[0, 0], scale=1, fill=0)
        tensor = torch.clamp(tensor,min=0.0,max=1.0)

        return tensor, (angle, None)