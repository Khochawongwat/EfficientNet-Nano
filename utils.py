from sympy import erfinv
import torch
from torch import nn
import numpy as np

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return SwishImplementation.apply(x, self.beta)
                                         
class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i, beta):
        result = i * torch.sigmoid(beta * i)
        ctx.save_for_backward(i, beta)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i, beta = ctx.saved_tensors
        sigmoid_i = torch.sigmoid(beta * i)
        return grad_output * (beta * sigmoid_i * (1 + i * (1 - sigmoid_i))), None