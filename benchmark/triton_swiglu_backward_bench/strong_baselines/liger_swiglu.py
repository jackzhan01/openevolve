"""Liger SwiGLU strong baseline wrappers.

Wraps liger_kernel.ops.swiglu to match the standalone-backward and
autograd-pair APIs used in this benchmark.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.distributed.tensor  # noqa: F401  # ensures torch.distributed.tensor.DTensor resolves in liger_kernel


def liger_available() -> bool:
    try:
        import liger_kernel.ops.swiglu  # noqa: F401
        return True
    except ImportError:
        return False


def make_liger_swiglu_backward_fn() -> Callable:
    """Standalone backward matching swiglu_backward_triton(dc, a, b) -> (da, db)."""
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    def swiglu_backward_liger(
        dc: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_req = a.detach().clone().requires_grad_(True)
        b_req = b.detach().clone().requires_grad_(True)
        c = LigerSiLUMulFunction.apply(a_req, b_req)
        torch.autograd.backward(c, dc.detach().clone())
        return a_req.grad, b_req.grad

    return swiglu_backward_liger


def make_liger_swiglu_train_step_fn() -> Callable:
    """Forward + backward full training step using LigerSiLUMulFunction."""
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    def train_step(
        a_req: torch.Tensor,
        b_req: torch.Tensor,
        dc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_req.grad = None
        b_req.grad = None
        c = LigerSiLUMulFunction.apply(a_req, b_req)
        c.backward(dc)
        return a_req.grad, b_req.grad

    return train_step


def make_liger_swiglu_autograd_pair_fns() -> tuple[Callable, Callable]:
    """Return (forward_with_saved, backward_from_saved) wrappers using Liger.

    These match the autograd-pair contract:
        swiglu_forward_with_saved(a, b) -> (c, saved_tensors)
        swiglu_backward_from_saved(dc, saved_tensors) -> (da, db)

    Liger's kernel recomputes sigmoid(a) in backward, so the saved tensors
    are just (a, b) — the full inputs.
    """
    from liger_kernel.ops.swiglu import swiglu_forward, swiglu_backward

    def forward_with_saved(a: torch.Tensor, b: torch.Tensor):
        a_save, b_save, c = swiglu_forward(a, b)
        return c, (a_save, b_save)

    def backward_from_saved(
        dc: torch.Tensor,
        saved_tensors: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_save, b_save = saved_tensors
        da, db = swiglu_backward(a_save, b_save, dc)
        return da, db

    return forward_with_saved, backward_from_saved
