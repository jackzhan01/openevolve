"""PyTorch reference for PolyNorm.

Forward:
    y = w0 * norm(x^3) + w1 * norm(x^2) + w2 * norm(x) + b
    where norm(u) = u * rsqrt(mean_row(u^2) + eps)   (row-wise RMS normalization)

matching Liger's `LigerPolyNormFunction` with eps pinned to 1e-6. The kernel computes in
fp32 and stores in the input dtype; we do the same.

Inputs:
    x      : [rows, cols]  (differentiable)
    weight : [3]  = [w0, w1, w2]  (differentiable)
    bias   : [1]  scalar bias     (differentiable)
"""

import torch


def poly_norm_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    w = weight.float()

    def _norm(u):
        return u * torch.rsqrt(u.pow(2).mean(dim=-1, keepdim=True) + eps)

    y = w[0] * _norm(xf**3) + w[1] * _norm(xf**2) + w[2] * _norm(xf) + bias.float()
    return y.to(x.dtype)
