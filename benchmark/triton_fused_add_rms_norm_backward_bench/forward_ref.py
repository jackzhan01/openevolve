"""PyTorch reference forward for fused_add_rms_norm (used for AtenIR extraction and as the oracle)."""

import torch

def fused_add_rms_norm_forward(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    s = x + r
    rstd = torch.rsqrt(s.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (s * rstd.to(s.dtype)) * weight


fused_add_rms_norm_forward_ref = fused_add_rms_norm_forward

