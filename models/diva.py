import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 均值
def mean_channels_h(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True)
    return spatial_sum / F.size(3)


def mean_channels_w(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(2, keepdim=True)
    return spatial_sum / F.size(2)

# 方差
def var_channels_h(F, if_mean=False):
    assert(F.dim() == 4)
    F_mean = mean_channels_h(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True) / F.size(3)
    if if_mean:
        return F_variance + F_mean
    else:
        return F_variance


def var_channels_w(F, if_mean=False):
    assert(F.dim() == 4)
    F_mean = mean_channels_w(F)
    F_variance = (F - F_mean).pow(2).sum(2, keepdim=True) / F.size(2)
    if if_mean:
        return F_variance + F_mean
    else:
        return F_variance

# 标准差
def stdv_channels_h(F, if_mean=False):
    assert(F.dim() == 4)
    F_mean = mean_channels_h(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True) / F.size(3)
    F_stdv = F_variance.sqrt()
    if if_mean:
        return F_stdv + F_mean
    else: 
        return F_stdv

def stdv_channels_w(F, if_mean=False):
    assert(F.dim() == 4)
    F_mean = mean_channels_w(F)
    F_variance = (F - F_mean).pow(2).sum(2, keepdim=True) / F.size(2)
    F_stdv = F_variance.sqrt()
    if if_mean:
        return F_stdv + F_mean
    else:
        return F_stdv 


class DiVA_attention(nn.Module):
    def __init__(self, embed_dim, esa_type='stdv'):
        super().__init__()

        if esa_type == 'stdv':
            self.contrast_h = stdv_channels_h
            self.contrast_w = stdv_channels_w
        elif esa_type == 'var':
            self.contrast_h = var_channels_h
            self.contrast_w = var_channels_w
        else:
            raise Exception('Please input correct esa type.')        

        self.conv_h = nn.Conv2d(embed_dim, embed_dim, 
                                kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(embed_dim, embed_dim, 
                                kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        n, c, h, w = x.size()

        identity = x

        c_h = self.contrast_h(x, if_mean=True)
        c_w = self.contrast_w(x, if_mean=True)

        a_h = self.conv_h(c_h).sigmoid()
        a_w = self.conv_w(c_w).sigmoid()

        out = identity * a_w * a_h

        return out
    