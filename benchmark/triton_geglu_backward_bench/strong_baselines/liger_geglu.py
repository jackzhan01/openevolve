"""Liger GeGLU strong-baseline wrappers.

Wraps ``liger_kernel.ops.geglu`` raw ``geglu_forward`` / ``geglu_backward``.
The raw backward writes gradients in-place into its ``a`` and ``b`` arguments, so the saved
forward tensors are cloned inside ``backward_from_saved`` to keep repeated backward calls pure.
"""

from __future__ import annotations

import importlib as _importlib

import torch as _torch
import torch.distributed.tensor as _torch_distributed_tensor  # noqa: F401

__all__ = ["liger_available", "make_liger_geglu_autograd_pair_fns"]

_LIGER_GEGLU_MODULE = "liger_kernel.ops.geglu"


def _load_liger_geglu_module():
    _mod = _importlib.import_module(_LIGER_GEGLU_MODULE)
    if not hasattr(_mod, "geglu_forward") or not hasattr(_mod, "geglu_backward"):
        raise ImportError(f"{_LIGER_GEGLU_MODULE} does not expose geglu_forward/geglu_backward")
    return _mod


def liger_available() -> bool:
    try:
        _load_liger_geglu_module()
        return True
    except ImportError:
        return False


def make_liger_geglu_autograd_pair_fns() -> tuple:
    _mod = _load_liger_geglu_module()
    _geglu_forward = _mod.geglu_forward
    _geglu_backward = _mod.geglu_backward

    def forward_with_saved(a: _torch.Tensor, b: _torch.Tensor):
        a = a.contiguous()
        b = b.contiguous()
        a_saved, b_saved, c = _geglu_forward(a, b)
        return c, (a_saved, b_saved)

    def backward_from_saved(dc: _torch.Tensor, saved_tensors):
        a_saved, b_saved = saved_tensors

        dc = dc.contiguous()

        # Liger's raw GeGLU backward stores da into ``a`` and db into ``b`` in-place.
        # Clone saved tensors so repeated calls with the same saved_tensors remain pure.
        a_work = a_saved.clone().contiguous()
        b_work = b_saved.clone().contiguous()

        da, db = _geglu_backward(a_work, b_work, dc)
        return da, db

    return forward_with_saved, backward_from_saved
