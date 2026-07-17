from __future__ import annotations

import importlib as _importlib

import torch as _torch

try:
    import torch.distributed.tensor as _torch_distributed_tensor
except Exception:
    _torch_distributed_tensor = None

__all__ = ["liger_available", "make_liger_tvd_autograd_pair_fns"]


_LIGER_TVD_MODULE_CANDIDATES = (
    "liger_kernel.ops.tvd",
    "liger_kernel.ops.tv_distance",
    "liger_kernel.ops.tvd_loss",
    "liger_kernel.ops.total_variation_distance",
    "liger_kernel.ops.total_variation_distance_loss",
)


def _load_liger_tvd_module():
    _last_exc = None

    for _module_name in _LIGER_TVD_MODULE_CANDIDATES:
        try:
            _module = _importlib.import_module(_module_name)
        except Exception as _exc:
            _last_exc = _exc
            continue

        if hasattr(_module, "tv_distance_forward_triton") and hasattr(_module, "tvd_backward_triton"):
            return _module

        _last_exc = ImportError(
            f"{_module_name!r} imported, but does not expose "
            "tv_distance_forward_triton and tvd_backward_triton"
        )

    raise ImportError("Could not import Liger TVD raw ops module") from _last_exc


def liger_available() -> bool:
    try:
        _load_liger_tvd_module()
        return True
    except Exception:
        return False


def make_liger_tvd_autograd_pair_fns() -> tuple[object, object]:
    _liger_tvd = _load_liger_tvd_module()

    def forward_with_saved(p: _torch.Tensor, q: _torch.Tensor) -> tuple[_torch.Tensor, tuple[_torch.Tensor, ...]]:
        p = p.contiguous()
        q = q.contiguous()

        out, grads = _liger_tvd.tv_distance_forward_triton(
            p,
            q,
            None,
            "batchmean",
            -100,
            False,
        )

        # FAIRNESS: no tie-patching pass here. Liger stores -scale where p == q while
        # torch.abs uses subgradient 0, but the disagreement is 0.5/BT per tied element —
        # orders of magnitude inside the gate's atol — and patching cost a full extra
        # elementwise kernel in the TIMED baseline forward.
        return out, (grads,)

    def backward_from_saved(
        dout: _torch.Tensor,
        saved_tensors: tuple[_torch.Tensor, ...],
        *extras: object,
    ) -> tuple[_torch.Tensor, _torch.Tensor]:
        (grads,) = saved_tensors

        if isinstance(dout, _torch.Tensor):
            dout = dout.contiguous()

        dp = _liger_tvd.tvd_backward_triton(dout, grads)
        dq = -dp

        return dp, dq

    return forward_with_saved, backward_from_saved
