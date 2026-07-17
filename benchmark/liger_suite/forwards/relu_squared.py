"""PyTorch reference for ReLU-squared activation.

Forward:
    y = relu(x) ** 2

matching Liger's `LigerReLUSquaredFunction` (element-wise, no parameters, computed in the
input dtype — no fp32 upcast in the Liger kernel).
"""

import torch


def relu_squared_forward(x: torch.Tensor) -> torch.Tensor:
    r = torch.relu(x)
    return r * r
