"""PyTorch reference forward for poly_norm (used for AtenIR extraction and as the oracle)."""

import torch

def poly_norm_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    w = weight.float()

    def _norm(u):
        return u * torch.rsqrt(u.pow(2).mean(dim=-1, keepdim=True) + eps)

    y = w[0] * _norm(xf**3) + w[1] * _norm(xf**2) + w[2] * _norm(xf) + bias.float()
    return y.to(x.dtype)


poly_norm_forward_ref = poly_norm_forward

