from __future__ import annotations

import importlib as _importlib

import torch as _torch

try:
    _importlib.import_module("torch.distributed.tensor")
except Exception:
    pass


__all__ = [
    "liger_available",
    "make_liger_fused_add_rms_norm_autograd_pair_fns",
]


_LIGER_MODULE_NAME = "liger_kernel.ops.fused_add_rms_norm"
_CASTING_MODE = "gemma"
_OFFSET = 0.0
_NUM_STAGES = 2
_LIGER_MODULE = None
_LIGER_IMPORT_ERROR = None


def _load_liger_module():
    global _LIGER_MODULE, _LIGER_IMPORT_ERROR

    if _LIGER_MODULE is not None:
        return _LIGER_MODULE
    if _LIGER_IMPORT_ERROR is not None:
        raise _LIGER_IMPORT_ERROR

    try:
        mod = _importlib.import_module(_LIGER_MODULE_NAME)
        getattr(mod, "fused_add_rms_norm_forward")
        getattr(mod, "fused_add_rms_norm_backward")
        getattr(mod, "calculate_settings")
        getattr(mod, "_str_to_casting_mode")
    except Exception as exc:
        _LIGER_IMPORT_ERROR = exc
        raise

    _LIGER_MODULE = mod
    return mod


def _casting_mode_value(mod):
    return mod._str_to_casting_mode[_CASTING_MODE]


def liger_available() -> bool:
    try:
        _load_liger_module()
        return True
    except Exception:
        return False


def make_liger_fused_add_rms_norm_autograd_pair_fns():
    mod = _load_liger_module()

    def forward_with_saved(x, r, weight, eps):
        x = x.contiguous()
        r = r.contiguous()
        weight = weight.contiguous()

        y, s, rstd, _block_size, _num_warps, _num_stages, _casting_mode = mod.fused_add_rms_norm_forward(
            x,
            r,
            weight,
            eps,
            _OFFSET,
            _CASTING_MODE,
        )

        saved_tensors = (s, weight, rstd)
        return y, saved_tensors

    def backward_from_saved(dout, saved_tensors, eps):
        if not isinstance(saved_tensors, tuple) or len(saved_tensors) != 3:
            raise RuntimeError("saved_tensors must be the flat tensor tuple returned by forward_with_saved")

        s, weight, rstd = saved_tensors
        if not (
            isinstance(s, _torch.Tensor)
            and isinstance(weight, _torch.Tensor)
            and isinstance(rstd, _torch.Tensor)
        ):
            raise RuntimeError("saved_tensors must contain tensors only")

        dout = dout.contiguous()
        s = s.contiguous()
        weight = weight.contiguous()
        rstd = rstd.contiguous()

        n_cols = s.shape[-1]
        block_size, num_warps = mod.calculate_settings(n_cols)
        casting_mode = _casting_mode_value(mod)

        d_s_out = _torch.zeros_like(s)

        dx, dr, dweight = mod.fused_add_rms_norm_backward(
            dout,
            d_s_out,
            s,
            weight,
            rstd,
            _OFFSET,
            casting_mode,
            block_size,
            num_warps,
            _NUM_STAGES,
            False,
        )

        return dx, dr, dweight

    return forward_with_saved, backward_from_saved
