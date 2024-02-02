import math
from sympy import erfinv
import torch
from torch import nn
import numpy as np
from scipy.special import erfinv
class Swish(nn.Module):
    def __init__(self, beta = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return SwishImplementation.apply(x, self.beta)          
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, beta):
        #Check if beta is a tensor because sometimes it is a float
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, dtype=i.dtype, device=i.device)
        result = i * torch.sigmoid(beta * i)
        ctx.save_for_backward(i, beta)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i, beta = ctx.saved_tensors
        sigmoid_i = torch.sigmoid(beta * i)
        return grad_output * (sigmoid_i * (1 + beta * i * (1 - sigmoid_i))), None

def normalize_tensor(y: torch.Tensor, eps: float):
    """
    Function to normalize a tensor.
    """
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor.")
    if not isinstance(eps, float):
        raise TypeError("eps must be a float.")
    
    return (y - torch.mean(y)) / torch.std(y)

def tensor_is_normalized(y: torch.Tensor, eps: float):
    """
    Function to check if a tensor is normalized.
    """
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor.")
    if not isinstance(eps, float):
        raise TypeError("eps must be a float.")
    
    return torch.allclose(torch.mean(y), torch.zeros(1, dtype=y.dtype), atol=eps) and \
        torch.allclose(torch.var(y), torch.ones(1, dtype=y.dtype), atol=eps)

def uniformly_sampled_gaussian(num_rand: int):
    """
    Function to generate uniformly sampled gaussian values.
    """
    if not isinstance(num_rand, int):
        raise TypeError("num_rand must be an integer.")
    
    rand = 2 * (np.arange(num_rand) + 0.5) / float(num_rand) - 1
    return np.sqrt(2) * erfinv(rand)


def create_channelwise_variable(y: torch.Tensor, init: float):
    """
    Function to create a channel-wise variable.
    """
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor.")
    if not isinstance(init, float):
        raise TypeError("init must be a float.")
    
    num_channels = int(y.shape[-1])
    return nn.Parameter(init * torch.ones((1, 1, 1, num_channels), dtype=y.dtype))

class ProxyNormalization(nn.Module): 
    """
    Proxy Normalization class.
    """
    def __init__(self, y: torch.Tensor, activation_fn: callable, eps: float = 0.03, n_samples: int = 256, apply_activation: bool = False):
        super().__init__()

        if(y.ndim != 4):
            raise ValueError("y must be a 4-dimensional tensor.")
        
        if y.shape == (1, 1, 1, 1):
            raise ValueError("y is a scalar. Proxy Normalization will not be applied.")
            
        if(not tensor_is_normalized(y, eps)):
            raise ValueError("y must be normalized.")
        
        self.y = y
        self.activation_fn = activation_fn
        self.eps = eps
        self.n_samples = n_samples
        self.apply_activation = apply_activation

    def forward(self) -> torch.Tensor:
        beta = create_channelwise_variable(self.y, 0.0)
        gamma = create_channelwise_variable(self.y, 1.0)
        
        #affine transform on y and apply activation function
        z = self.activation_fn((gamma * self.y) + beta)

        #Create a proxy distribution of y
        proxy_y = torch.tensor(uniformly_sampled_gaussian(self.n_samples), dtype=self.y.dtype)

        proxy_y = torch.reshape(proxy_y, (self.n_samples, 1, 1, 1))
        
        #Affine transform on proxy of y and apply activation function
        proxy_z = self.activation_fn((gamma * proxy_y) + beta)
        proxy_z = proxy_z.type(self.y.dtype)
        
        proxy_mean = torch.mean(proxy_z, dim=0, keepdim=True)
        proxy_var = torch.var(proxy_z, dim=0, unbiased=False, keepdim=True)
        
        proxy_mean = proxy_mean.type(self.y.dtype)
        
        inv_proxy_std = torch.rsqrt(proxy_var + self.eps)
        inv_proxy_std = inv_proxy_std.type(self.y.dtype)

        tilde_z = (z - proxy_mean)  * inv_proxy_std

        if self.apply_activation:
            tilde_z = self.activation_fn(tilde_z)

        return tilde_z
    
def drop_connect(x, drop_connect_rate, training):
    if not training:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.shape[0]
    random_tensor = torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    random_tensor.add_(keep_prob)
    binary_mask = torch.floor(random_tensor)
    reci_keep_prob = 1.0 / keep_prob
    x = x * binary_mask * reci_keep_prob
    return x

def round_filters(filters, multiplier = None, divisor=8, min_width=None):
    if not multiplier:
        return filters
    if multiplier != 1:
        filters *= multiplier
    min_width = min_width or divisor
    new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, multiplier = None):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))