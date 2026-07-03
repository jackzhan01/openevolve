"""Autograd-pair matmul (GEMM) seed with an evolvable saved-tensor contract.

Public contract:

    matmul_forward_with_saved(a, b) -> (c, saved_tensors)
    matmul_backward_from_saved(dc, saved_tensors) -> (da, db)

The initial seed is deliberately conservative: it only saves the original
forward inputs needed by a standalone backward.  There is no eps for plain GEMM.
"""

from __future__ import annotations

import torch


def matmul_forward_torch(a, b):
    return a @ b


def matmul_backward_torch(dc, a, b):
    da = (dc.float() @ b.float().T).to(a.dtype)
    db = (a.float().T @ dc.float()).to(b.dtype)
    return da, db


# EVOLVE-BLOCK-START
from backward_naive_triton import matmul_backward_naive_triton as _seed_backward
from forward_triton import matmul_forward_triton as _seed_forward


def _forward_with_saved_impl(a, b):
    """Initial evolvable forward: save only original inputs.

    OpenEvolve may add forward-computed intermediates to this tuple and update
    ``_backward_from_saved_impl`` to consume them.
    """
    c = _seed_forward(a, b)
    return c, (a.contiguous(), b.contiguous())


def _backward_from_saved_impl(dc, saved_tensors):
    """Initial evolvable backward: standalone backward over saved inputs."""
    a, b = saved_tensors[:2]
    return _seed_backward(dc, a, b)
# EVOLVE-BLOCK-END


def matmul_forward_with_saved(a: torch.Tensor, b: torch.Tensor):
    """Return forward output plus an evolvable saved tensor tuple."""
    if not (a.is_cuda and b.is_cuda):
        c = matmul_forward_torch(a, b)
        return c, (a, b)
    return _forward_with_saved_impl(a, b)


def matmul_backward_from_saved(dc: torch.Tensor, saved_tensors):
    """Consume saved tensors and return ``da, db``."""
    a, b = saved_tensors[:2]
    if not (dc.is_cuda and a.is_cuda and b.is_cuda):
        return matmul_backward_torch(dc, a, b)
    return _backward_from_saved_impl(dc, saved_tensors)
