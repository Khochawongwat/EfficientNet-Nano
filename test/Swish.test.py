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

    def test_forward_beta(self):
        for beta in [0.0, 0.5, 1.0, 2.0]:
            swish = Swish(beta=beta)
            x = torch.randn(10, requires_grad=True, dtype=torch.float64)  # gradcheck requires float64
            y = swish(x)
            expected_y = x * torch.sigmoid(beta * x)
            self.assertTrue(torch.allclose(y, expected_y), f"Swish forward pass is incorrect for beta={beta}.")
    
    def test_backward_beta(self):
        for beta in [0.0, 0.5, 1.0, 2.0]:
            swish = Swish(beta=beta)
            x = torch.randn(10, requires_grad=True, dtype=torch.float64)
            self.assertTrue(gradcheck(swish, (x,)), f"Swish backward pass is incorrect for beta={beta}.")
    
    def test_forward_shapes(self):
        swish = Swish(beta=1.0)
        x = torch.randn(10, 20, 30, requires_grad=True)
        y = swish(x)
        self.assertEqual(y.shape, x.shape, "Swish forward pass returns incorrect shape.")
    
    def test_backward_shapes(self):
        swish = Swish(beta=1.0)
        for shape in [(10,), (10, 10), (10, 10, 10)]:
            x = torch.randn(*shape, requires_grad=True, dtype=torch.float64)
            self.assertTrue(gradcheck(swish, (x,)), f"Swish backward pass is incorrect for shape={shape}.")            
class TestSwishImplementation(unittest.TestCase):
    def test_forward(self):
        swish_impl = SwishImplementation.apply
        x = torch.randn(10, requires_grad=True)
        beta = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        y = swish_impl(x, beta)
        expected_y = x * torch.sigmoid(beta * x)
        self.assertTrue(torch.allclose(y, expected_y), "SwishImplementation forward pass is incorrect.")
    
    def test_backward(self):
        swish_impl = SwishImplementation.apply
        x = torch.randn(10, requires_grad=True, dtype=torch.float64)
        beta = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        self.assertTrue(gradcheck(swish_impl, (x, beta)), "SwishImplementation backward pass is incorrect.")

    def test_forward_beta(self):
        swish_impl = SwishImplementation.apply
        x = torch.randn(10, requires_grad=True)
        for beta in [0.0, 0.5, 1.0, 2.0]:
            beta = torch.tensor(beta, dtype=x.dtype, device=x.device)
            y = swish_impl(x, beta)
            expected_y = x * torch.sigmoid(beta * x)
            self.assertTrue(torch.allclose(y, expected_y), f"SwishImplementation forward pass is incorrect for beta={beta.item()}.")

    def test_backward_beta(self):
        swish_impl = SwishImplementation.apply
        x = torch.randn(10, requires_grad=True, dtype=torch.float64)  # gradcheck requires float64
        for beta in [0.0, 0.5, 1.0, 2.0]:
            beta = torch.tensor(beta, dtype=x.dtype, device=x.device)
            self.assertTrue(gradcheck(swish_impl, (x, beta)), f"SwishImplementation backward pass is incorrect for beta={beta.item()}.")

    def test_forward_shapes(self):
        swish_impl = SwishImplementation.apply
        x = torch.randn(10, 20, 30, requires_grad=True)
        beta = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        y = swish_impl(x, beta)
        self.assertEqual(y.shape, x.shape, "SwishImplementation forward pass returns incorrect shape.") 
    
    def test_backward_shapes(self):
        swish_impl = SwishImplementation.apply
        beta = torch.tensor(1.0, dtype=torch.float64)
        for shape in [(10,), (10, 10), (10, 10, 10)]:
            x = torch.randn(*shape, requires_grad=True, dtype=torch.float64)  # gradcheck requires float64
            self.assertTrue(gradcheck(swish_impl, (x, beta)), f"SwishImplementation backward pass is incorrect for shape={shape}.")

if __name__ == '__main__':
    unittest.main()