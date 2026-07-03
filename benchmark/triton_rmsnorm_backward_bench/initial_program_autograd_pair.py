"""Autograd-pair RMSNorm seed with an evolvable saved-tensor contract.

Public contract:

    rmsnorm_forward_with_saved(x, weight, eps) -> (y, saved_tensors)
    rmsnorm_backward_from_saved(dy, saved_tensors, eps) -> (dx, dweight)

The initial seed is deliberately conservative: it only saves the original
forward inputs needed by a standalone backward.  OpenEvolve is expected to
modify the EVOLVE-BLOCK if saving additional forward intermediates (such as the
per-row reciprocal RMS) improves the forward+backward tradeoff.
"""

from __future__ import annotations

import torch


def rmsnorm_forward_torch(x, weight, eps=1e-5):
    xf = x.float()
    ms = (xf * xf).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(ms + float(eps))
    y = (xf * rrms * weight.float()).to(x.dtype)
    return y


def rmsnorm_backward_torch(dy, x, weight, eps=1e-5):
    xf = x.float()
    dyf = dy.float()
    wf = weight.float()
    ms = (xf * xf).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(ms + float(eps))
    xhat = xf * rrms
    g = dyf * wf
    dx = (g - xhat * (g * xhat).mean(dim=-1, keepdim=True)) * rrms
    dweight = (dyf * xhat).sum(dim=0).to(weight.dtype)
    return dx.to(x.dtype), dweight


# EVOLVE-BLOCK-START
from backward_naive_triton import rmsnorm_backward_naive_triton as _seed_backward
from forward_triton import rmsnorm_forward_triton as _seed_forward


def _forward_with_saved_impl(x, weight, eps=1e-5):
    """Initial evolvable forward: save only original inputs.

    OpenEvolve may add forward-computed intermediates to this tuple and update
    ``_backward_from_saved_impl`` to consume them.
    """
    y = _seed_forward(x, weight, eps)
    return y, (x.contiguous(), weight.contiguous())


def _backward_from_saved_impl(dy, saved_tensors, eps=1e-5):
    """Initial evolvable backward: standalone backward over saved inputs."""
    x, weight = saved_tensors[:2]
    return _seed_backward(dy, x, weight, eps)
# EVOLVE-BLOCK-END


def rmsnorm_forward_with_saved(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
):
    """Return forward output plus an evolvable saved tensor tuple."""
    if not (x.is_cuda and weight.is_cuda):
        y = rmsnorm_forward_torch(x, weight, eps)
        return y, (x, weight)
    return _forward_with_saved_impl(x, weight, eps)


def rmsnorm_backward_from_saved(
    dy: torch.Tensor,
    saved_tensors,
    eps: float = 1e-5,
):
    """Consume saved tensors and return ``dx, dweight``."""
    x, weight = saved_tensors[:2]
    if not (dy.is_cuda and x.is_cuda and weight.is_cuda):
        return rmsnorm_backward_torch(dy, x, weight, eps)
    return _backward_from_saved_impl(dy, saved_tensors, eps)
