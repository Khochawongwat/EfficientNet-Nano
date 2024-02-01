import math
import torch
from torch import nn

from utils import *

params = {
    "efficientnet-nano": [1, 1, 224, 0.2],
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_size, kernel_size, stride, activation_fn, bias = True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=group_size, bias=bias)
        self.silu = Swish() if activation_fn else nn.Identity()

    def forward(self, x):
        return self.silu(self.conv(x))
    
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, include_se, proxy_norm):
        super().__init__()

        self.momentum = 0.01
        self.eps = 1e-3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.include_se = include_se
        self.proxy_norm = proxy_norm

        self.relu = nn.ReLU6(inplace = True)

        #Stages -> Expansion, Squeeze and Excitation, Depthwise Convolution, Projection
        
        #Expansion
        expansion_channels = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = ConvBlock(in_channels, expansion_channels, 1, 1, 1, activation_fn = True, bias=False)
            self.bn0 = nn.BatchNorm2d(expansion_channels, momentum=self.momentum, eps=self.eps)

        #Squeeze and Excitation
        if include_se:
            n_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se_reduce = ConvBlock(in_channels, n_squeezed_channels, 1, 1, 1, activation_fn = True)
            self.se_expand = ConvBlock(n_squeezed_channels, out_channels, 1, 1, 1, activation_fn = True)
        
        #Depthwise Convolution
            self.depthwise_conv = ConvBlock(expansion_channels, expansion_channels, expansion_channels, kernel_size, stride, activation_fn = True, bias = True)

        #Batch Normalization / Proxy Normalization
        if proxy_norm:
            pass
        else:
            self.bn1 = nn.BatchNorm2d(expansion_channels, momentum=self.momentum, eps=self.eps)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=self.momentum, eps=self.eps)
        
        #Projection
        self.project_conv = ConvBlock(expansion_channels, out_channels, 1, 1, 1, activation_fn = False, bias = False)