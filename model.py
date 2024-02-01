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
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, include_se = True, proxy_norm = True, group_size = 16, se_ratio = 0.25):
        super().__init__()

        self.momentum = 0.01
        self.eps = 1e-3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        #Expand_ratio to 4 instead of 6 because we are using grouped convolutions with group_size = 16 which could cause large memory consumption if it were to be kept at DEFAULT.
        self.expand_ratio = 1.0 if expand_ratio == 1.0 else 4.0

        self.se_ratio = se_ratio
        self.include_se = include_se
        self.proxy_norm = proxy_norm
        self.group_size = group_size
        self.activation_func = Swish()

        #Stages -> Expansion, Squeeze and Excitation, Depthwise Convolution, Projection
        
        #Expansion
        expansion_channels = in_channels * self.expand_ratio

        if self.expand_ratio != 1:
            self.expand_conv = ConvBlock(in_channels, expansion_channels, self.group_size,  1, 1, activation_fn = True, bias=False)
            self.bn0 = nn.BatchNorm2d(expansion_channels, momentum=self.momentum, eps=self.eps)

        #Squeeze and Excitation
        if include_se:
            n_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se_reduce = ConvBlock(in_channels, n_squeezed_channels, self.group_size, 1, 1, activation_fn = True)
            self.se_expand = ConvBlock(n_squeezed_channels, out_channels, self.group_size, 1, 1, activation_fn = True)
        
        #Depthwise Convolution
            self.depthwise_conv = ConvBlock(expansion_channels, expansion_channels, self.group_size, kernel_size, stride, activation_fn = True, bias = True)

        #Batch Normalization
        self.bn1 = nn.BatchNorm2d(expansion_channels, momentum=self.momentum, eps=self.eps)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=self.momentum, eps=self.eps)

        #Projection
        self.project_conv = ConvBlock(expansion_channels, out_channels, self.group_size, 1, 1, activation_fn = False, bias = False)

    def forward(self, x, drop_connect_rate = None):

        if drop_connect_rate and (drop_connect_rate > 1 or drop_connect_rate < 0):
            raise ValueError("drop_connect_rate must be between 0 and 1.")
        
        #Stages -> Reserve Identity Block, Expansion, Depthwise Convolution, Squeeze and Excitation, Projection, Residual Connection
        
        #Reserve Identity Block
        identity_block = x

        #Expansion
        if self.expand_ratio != 1:
            bn0 = self.bn0(self.expand_conv(x))
            if self.proxy_norm:
                #Usually an activation function is applied after batch normalization but here ProxyNormalization automatically applies the activation function to the normalized tensor, given apply_activation = True 
                x = ProxyNormalization(bn0, activation_fn = self.activation_func, eps = 0.03, n_samples = min(bn0.shape[0], 256), apply_activation = True).forward()
            else:
                x = self.activation_func(bn0)

        #Depthwise Convolution
        bn1 = self.bn1(self.depthwise_conv(x))
        if self.proxy_norm:
            x = ProxyNormalization(bn1, activation_fn = self.activation_func, eps = 0.03, n_samples =  min(bn1.shape[0], 256), apply_activation = True).forward()
        else:
            x = self.activation_func(bn1)
        
        #Squeeze and Excitation
        if self.include_se:
            squeezed_x = nn.AdaptiveAvgPool1d(x, 1)
            squeezed_x = self.se_expand(self.activation_func(self.se_reduce(squeezed_x)))

            #Activate the squeezed_x for output range of 0 to 1
            x = torch.sigmoid(squeezed_x) * x

        #Projection
        bn2 = self.bn2(self.project_conv(x))
        if self.proxy_norm:
            x = ProxyNormalization(bn2, activation_fn = self.activation_func, eps = 0.03, n_samples =  min(bn2.shape[0], 256), apply_activation = True).forward()
        else:
            x = self.activation_func(bn2)

        # Add the identity block to the output of the projection if the input
        # and output channels are the same and the stride is 1
            
        if self.in_channels == self.out_channels and self.stride == 1:
            if drop_connect_rate:
                x = x + drop_connect(x, drop_connect_rate, training = self.mode == "train")
            x += identity_block

        return x


class EfficientNetNano(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self):
        pass