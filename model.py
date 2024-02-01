from torch import nn
from utils import *

params = {
    "efficientnet-nano": [1, 1, 224, 0.2],
}

def drop_connect(x, drop_connect_rate, training):
    if not training:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    x = (x / keep_prob) * binary_mask
    return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_size, kernel_size, stride, activation_fn, bias = True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=group_size, bias=bias)
        self.silu = Swish() if activation_fn else nn.Identity()

    def forward(self, x):
        return self.silu(self.conv(x))
    
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, se_ratio, include_se, proxy_norm):
        super().__init__()

        self.momentum = 0.01
        self.eps = 1e-3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expand_ratio = 4
        self.se_ratio = se_ratio
        self.include_se = include_se
        self.proxy_norm = proxy_norm

        self.relu = nn.ReLU6(inplace = True)

        #Stages -> Expansion, Squeeze and Excitation, Depthwise Convolution, Projection
        
        #Expansion
        expansion_channels = in_channels * self.expand_ratio

        if self.expand_ratio != 1:
            self.expand_conv = ConvBlock(in_channels, expansion_channels, 16, 1, 1, activation_fn = True, bias=False)
            self.bn0 = nn.BatchNorm2d(expansion_channels, momentum=self.momentum, eps=self.eps)

        #Squeeze and Excitation
        if include_se:
            n_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se_reduce = ConvBlock(in_channels, n_squeezed_channels, 16, 1, 1, activation_fn = True)
            self.se_expand = ConvBlock(n_squeezed_channels, out_channels, 16, 1, 1, activation_fn = True)
        
        #Depthwise Convolution
            self.depthwise_conv = ConvBlock(expansion_channels, expansion_channels, 16, kernel_size, stride, activation_fn = True, bias = True)

        #Batch Normalization
        self.bn1 = nn.BatchNorm2d(expansion_channels, momentum=self.momentum, eps=self.eps)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=self.momentum, eps=self.eps)

        #Projection
        self.project_conv = ConvBlock(expansion_channels, out_channels, 16, 1, 1, activation_fn = False, bias = False)

    def forward(self, x, drop_connect_rate = None):

        if drop_connect_rate and (drop_connect_rate > 1 or drop_connect_rate < 0):
            raise ValueError("drop_connect_rate must be between 0 and 1.")
        
        #Stages -> Reserve Identity Block, Expansion, Depthwise Convolution, Squeeze and Excitation, Projection, Residual Connection
        
        #Reserve Identity Block
        identity_block = x

        #Expansion
        if self.expand_ratio != 1:
            bn0 = self.bn0(self.expand_conv(x))
            x = ProxyNormalization(bn0, activation_fn = self.relu, eps = 0.03, n_samples = min(bn0.shape[0], 256), apply_activation = True).forward()

        #Depthwise Convolution
        bn1 = self.bn1(self.depthwise_conv(x))
        x = ProxyNormalization(bn1, activation_fn = self.relu, eps = 0.03, n_samples =  min(bn1.shape[0], 256), apply_activation = True).forward()
        
        #Squeeze and Excitation
        if self.include_se:
            squeezed_x = nn.AdaptiveAvgPool1d(x, 1)
            squeezed_x = self.se_expand(self.relu(self.se_reduce(squeezed_x)))

            #Activate the squeezed_x for output range of 0 to 1
            x = torch.sigmoid(squeezed_x) * x

        #Projection
        bn2 = self.bn2(self.project_conv(x))
        x = ProxyNormalization(bn2, activation_fn = self.relu, eps = 0.03, n_samples =  min(bn2.shape[0], 256), apply_activation = True).forward()

        # Add the identity block to the output of the projection if the input
        # and output channels are the same and the stride is 1
            
        if self.in_channels == self.out_channels and self.stride == 1:
            if drop_connect_rate:
                x = x + drop_connect(x, drop_connect_rate, training = self.mode == "train")
            x += identity_block

        return x