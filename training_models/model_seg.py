import math
import random
from collections import namedtuple

import torch
from torch import nn as nn
import torch.nn.functional as F

from util.logconf import logging
from util.unet import UNet

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()
        self._init_weights()
    def _init_weights(self): # use as boiler plate
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_( # works good for ReLU
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, 0, bound)

        
    def forward(self,input_batch):
        bn_out = self.input_batchnorm(input_batch)
        un_out = self.unet(bn_out)
        fn_out = self.final(un_out)
        return fn_out

class SegmentationAugmentation(nn.Module):
    """we transferred the augmentation to nn.Modude from dataset.
    as here the matrices will be tranferred to gpu for parallel computation
    and the affine and grid sample functions are more optimised for gpu,
    moreover, this augmentation was acting as a bottleneck previously
    
    on DAtaset : Disk → CPU load → CPU augment → send to GPU → train
    on nn.Module : Disk → CPU load → send to GPU → augment + train (GPU)
    """
    def __init__(
            self,
            flip = None,
            offset = None,
            scale = None,
            rotate = None,
            noise = None,
    ):
        super().__init__()
        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise
        
    def forward(self,input_g,label_g):
    # here label_g is the the ground-truth segmentation mask for the current sample

        transform_t = self._build2dTransformMatrix()
        transform_t = transform_t.expand(input_g.shape[0],-1,-1) # its expands the dimension of the matrix, kinda like unsqueeze
        transform_t = transform_t.to(input_g.device,torch.float32)
        affine_t = torch.nn.functional.affine_grid(transform_t[:,:2],input_g.size(),align_corners=False)
        # we did transform_t[:,:2] ,as the first dimesion is batch size and the now are dealing with 2d images

        augmented_input_g = torch.nn.functional.grid_sample(input_g,
                                                            affine_t,padding_mode='border',
                                                            align_corners=False)
        augmented_label_g = torch.nn.functional.grid_sample(label_g,
                                                            affine_t,padding_mode='border',
                                                            align_corners=False)
        # why are we also augmenting label ?
         # Input (CT)         Label (mask)
        #        tumor  ←→  tumor mask

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise
            augmented_input_g += noise_t

        return augmented_input_g,(augmented_label_g>0.5 ).to(torch.float32) # taking 0.5 as the threshold 
    


    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i,i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[i, 2] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i,i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            transform_t @= rotation_t

        return transform_t