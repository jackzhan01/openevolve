"""Autograd-pair Linear seed with an evolvable saved-tensor contract.

Public contract:

    linear_forward_with_saved(x, weight, bias) -> (y, saved_tensors)
    linear_backward_from_saved(dy, saved_tensors) -> (dx, dweight, dbias)

The initial seed is deliberately conservative: it only saves the original
forward inputs (x, weight) needed by a standalone backward.  dbias = sum(dy) does
not depend on the bias value, so bias is not saved.  There is no eps for Linear.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear_forward_torch(x, weight, bias):
    return F.linear(x, weight, bias)


def linear_backward_torch(dy, x, weight):
    dyf = dy.float()
    dx = (dyf @ weight.float()).to(x.dtype)
    dweight = (dyf.T @ x.float()).to(weight.dtype)
    dbias = dyf.sum(dim=0).to(dy.dtype)
    return dx, dweight, dbias


# EVOLVE-BLOCK-START
from backward_naive_triton import linear_backward_naive_triton as _seed_backward
from forward_triton import linear_forward_triton as _seed_forward


def _forward_with_saved_impl(x, weight, bias):
    """Initial evolvable forward: save only the inputs the backward needs.

    OpenEvolve may add forward-computed intermediates to this tuple and update
    ``_backward_from_saved_impl`` to consume them.
    """
    y = _seed_forward(x, weight, bias)
    return y, (x.contiguous(), weight.contiguous())


def _backward_from_saved_impl(dy, saved_tensors):
    """Initial evolvable backward: standalone backward over saved inputs."""
    x, weight = saved_tensors[:2]
    return _seed_backward(dy, x, weight)
# EVOLVE-BLOCK-END


def linear_forward_with_saved(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """Return forward output plus an evolvable saved tensor tuple."""
    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        y = linear_forward_torch(x, weight, bias)
        return y, (x, weight)
    return _forward_with_saved_impl(x, weight, bias)


def linear_backward_from_saved(dy: torch.Tensor, saved_tensors):
    """Consume saved tensors and return ``dx, dweight, dbias``."""
    x, weight = saved_tensors[:2]
    if not (dy.is_cuda and x.is_cuda and weight.is_cuda):
        return linear_backward_torch(dy, x, weight)
    return _backward_from_saved_impl(dy, saved_tensors)
