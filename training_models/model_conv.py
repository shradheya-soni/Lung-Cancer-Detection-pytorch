#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import os
# get_ipython().run_line_magic('cd', '..')
# os.getcwd()


# In[4]:


import math

from torch import nn as nn

from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG) # the order is warning -> info -> debug



# In[ ]:


class LunaModel(nn.Module):
    def __init__(self,in_channels = 1,out_channels = 8):
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(1)
        self.block1 = Block(in_channels,out_channels)
        self.block2 = Block(out_channels,2*out_channels)
        self.block3 = Block(2*out_channels,4*out_channels)
        self.block4 = Block(4*out_channels,8*out_channels)

        self.head_linear = nn.Linear(1152,2)
        self.head_softmax = nn.Softmax(dim=1)

        
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
                    nn.init.normal_(m.bias, -bound, bound)

        
    def forward(self,x):
        x = self.tail_batchnorm(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        conv_flat = x.view(x.size(0),-1) # converts the size of (8,64,3,3,2) to (8,1152) to match the linear layer
        x = self.head_linear(conv_flat)
        
        return x,self.head_softmax(x)
        

class Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels,out_channels,padding =1 ,kernel_size = 3, bias = True)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(out_channels,out_channels,padding =1 ,kernel_size = 3, bias = True)
        self.relu2 = nn.ReLU(inplace = True)
        self.max_pool = nn.MaxPool3d(2,2)
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool(x)

        return x
        
        

