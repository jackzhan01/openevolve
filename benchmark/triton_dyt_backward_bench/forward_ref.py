"""PyTorch reference forward for dyt (used for AtenIR extraction and as the oracle)."""

import torch

def dyt_forward(x: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    y = gamma.float() * torch.tanh(alpha.float() * x.float()) + beta.float()
    return y.to(x.dtype)


dyt_forward_ref = dyt_forward

