import unittest
import torch
from torch.autograd import gradcheck

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import Swish, SwishImplementation

class TestSwish(unittest.TestCase):
    def test_forward(self):
        swish = Swish(beta=1.0)
        x = torch.randn(10, requires_grad=True, dtype=torch.float64)  # gradcheck requires float64
        y = swish(x)
        expected_y = x * torch.sigmoid(x)
        self.assertTrue(torch.allclose(y, expected_y), "Swish forward pass is incorrect.")

    def test_backward(self):
        swish = Swish(beta=1.0)
        x = torch.randn(10, requires_grad=True, dtype=torch.float64)  # gradcheck requires float64
        self.assertTrue(gradcheck(swish, (x,)), "Swish backward pass is incorrect.")

class TestSwishImplementation(unittest.TestCase):
    def test_forward(self):
        swish_impl = SwishImplementation.apply
        x = torch.randn(10, requires_grad=True)
        beta = torch.tensor(1.0)
        y = swish_impl(x, beta)
        expected_y = x * torch.sigmoid(beta * x)
        self.assertTrue(torch.allclose(y, expected_y), "SwishImplementation forward pass is incorrect.")

    def test_backward(self):
        swish_impl = SwishImplementation.apply
        x = torch.randn(10, requires_grad=True, dtype=torch.float64)  # gradcheck requires float64
        beta = torch.tensor(1.0, dtype=torch.float64)
        self.assertTrue(gradcheck(swish_impl, (x, beta)), "SwishImplementation backward pass is incorrect.")


if __name__ == '__main__':
    unittest.main()