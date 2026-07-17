"""PyTorch reference for fused residual-add + RMSNorm (single-output form).

Forward:
    s = x + r                      # residual add (the "fused add")
    y = s * rsqrt(mean_row(s^2) + eps) * weight

matching Liger's `LigerFusedAddRMSNormFunction` with the pinned config: offset=0.0,
casting_mode="llama" (only the inverse RMS in fp32), eps=1e-6.

NOTE: Liger's kernel returns BOTH (y, s) — the updated residual `s` feeds the next decoder
layer. This benchmark case uses the single-output form (y only); gradients w.r.t. x, r and
weight are still fully defined through y. If the auto Liger wrapper cannot bridge the
two-output interface fairly, this op falls back to a hand-written baseline.

Inputs:
    x      : [rows, cols]  hidden states (differentiable)
    r      : [rows, cols]  residual      (differentiable)
    weight : [cols]                       (differentiable)
"""

import torch


def fused_add_rms_norm_forward(x: torch.Tensor, r: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    s = x + r
    rstd = torch.rsqrt(s.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (s * rstd.to(s.dtype)) * weight
