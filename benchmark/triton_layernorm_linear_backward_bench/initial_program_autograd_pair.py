"""Autograd-pair LayerNorm -> Linear seed with an evolvable saved-tensor contract.

Public contract:

    layernorm_linear_forward_with_saved(x, weight, bias, linear_weight, eps)
        -> (out, saved_tensors)
    layernorm_linear_backward_from_saved(dout, saved_tensors, eps)
        -> (dx, dlinear_weight, dweight, dbias)

``weight``/``bias`` are the LayerNorm affine params (gamma, beta); ``linear_weight``
is the Linear matrix B [K, N].  The initial seed is deliberately conservative: it
saves only the original forward inputs needed by a standalone backward.
OpenEvolve is expected to modify the EVOLVE-BLOCK if saving forward intermediates
(per-row mean/rstd, x_hat, or y_hat) improves the forward+backward tradeoff.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def layernorm_linear_forward_torch(x, weight, bias, linear_weight, eps=1e-5):
    k = x.shape[-1]
    y_hat = F.layer_norm(x.float(), (k,), weight.float(), bias.float(), float(eps))
    out = (y_hat @ linear_weight.float()).to(x.dtype)
    return out


def layernorm_linear_backward_torch(dout, x, weight, bias, linear_weight, eps=1e-5):
    k = x.shape[-1]
    x_ref = x.detach().float().requires_grad_(True)
    w_ref = weight.detach().float().requires_grad_(True)
    b_ref = bias.detach().float().requires_grad_(True)
    lw_ref = linear_weight.detach().float().requires_grad_(True)
    y_hat = F.layer_norm(x_ref, (k,), w_ref, b_ref, float(eps))
    out = y_hat @ lw_ref
    out.backward(dout.detach().float())
    return (
        x_ref.grad.to(x.dtype),
        lw_ref.grad.to(linear_weight.dtype),
        w_ref.grad.to(weight.dtype),
        b_ref.grad.to(bias.dtype),
    )


# EVOLVE-BLOCK-START
from backward_naive_triton import layernorm_linear_backward_naive_triton as _seed_backward
from forward_ref import layernorm_linear_forward_ref as _seed_forward


def _forward_with_saved_impl(x, weight, bias, linear_weight, eps=1e-5):
    """Initial evolvable forward: save only the original inputs.

    OpenEvolve may add forward-computed intermediates (mean/rstd, x_hat, y_hat)
    to this tuple and update ``_backward_from_saved_impl`` to consume them.
    """
    out = _seed_forward(x, weight, bias, linear_weight, eps)
    return out, (
        x.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        linear_weight.contiguous(),
    )


def _backward_from_saved_impl(dout, saved_tensors, eps=1e-5):
    """Initial evolvable backward: standalone backward over saved inputs."""
    x, weight, bias, linear_weight = saved_tensors[:4]
    return _seed_backward(dout, x, weight, bias, linear_weight, eps)
# EVOLVE-BLOCK-END


def layernorm_linear_forward_with_saved(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-5,
):
    """Return forward output plus an evolvable saved tensor tuple."""
    if not (x.is_cuda and weight.is_cuda and bias.is_cuda and linear_weight.is_cuda):
        out = layernorm_linear_forward_torch(x, weight, bias, linear_weight, eps)
        return out, (x, weight, bias, linear_weight)
    return _forward_with_saved_impl(x, weight, bias, linear_weight, eps)


def layernorm_linear_backward_from_saved(
    dout: torch.Tensor,
    saved_tensors,
    eps: float = 1e-5,
):
    """Consume saved tensors and return ``dx, dlinear_weight, dweight, dbias``."""
    x, weight, bias, linear_weight = saved_tensors[:4]
    if not (dout.is_cuda and x.is_cuda and weight.is_cuda and bias.is_cuda and linear_weight.is_cuda):
        return layernorm_linear_backward_torch(dout, x, weight, bias, linear_weight, eps)
    return _backward_from_saved_impl(dout, saved_tensors, eps)
