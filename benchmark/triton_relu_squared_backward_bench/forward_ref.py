"""PyTorch reference forward for relu_squared (used for AtenIR extraction and as the oracle)."""

import torch

def relu_squared_forward(x: torch.Tensor) -> torch.Tensor:
    r = torch.relu(x)
    return r * r


relu_squared_forward_ref = relu_squared_forward

