"""PyTorch reference for DyT (Dynamic Tanh, "Transformers without Normalization").

Forward:
    y = gamma * tanh(alpha * x) + beta

matching Liger's `LigerDyTFunction` with HAVE_BETA=True: alpha is a learnable SCALAR
(shape (1,)), gamma/beta are per-column vectors (shape (cols,)). Liger computes in
fp32 and stores in the input dtype; we do the same.

All four inputs are differentiable.
"""

import torch


def dyt_forward(x: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    y = gamma.float() * torch.tanh(alpha.float() * x.float()) + beta.float()
    return y.to(x.dtype)
