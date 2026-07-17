import importlib as _importlib

import torch as _torch

try:
    import torch.distributed.tensor as _torch_distributed_tensor
except Exception:
    _torch_distributed_tensor = None


def _import_liger_dyt_ops():
    _mod = _importlib.import_module("liger_kernel.ops.dyt")
    _fwd = getattr(_mod, "liger_dyt_fwd")
    _bwd = getattr(_mod, "liger_dyt_bwd")
    return _fwd, _bwd


def liger_available() -> bool:
    try:
        _import_liger_dyt_ops()
        return True
    except Exception:
        return False


def make_liger_dyt_autograd_pair_fns() -> tuple:
    try:
        _liger_dyt_fwd, _liger_dyt_bwd = _import_liger_dyt_ops()
    except Exception as _exc:
        raise ImportError("Liger DyT raw ops are not available") from _exc

    def forward_with_saved(x, alpha, gamma, beta):
        x = x.contiguous()
        alpha = alpha.contiguous()
        gamma = gamma.contiguous()

        if beta is None:
            beta_for_liger = None
            beta_saved = _torch.empty((0,), device=x.device, dtype=x.dtype)
        else:
            beta_for_liger = beta.contiguous()
            beta_saved = beta_for_liger

        y = _liger_dyt_fwd(x, alpha, gamma, beta_for_liger)
        saved_tensors = (x, alpha, gamma, beta_saved)
        return y, saved_tensors

    def backward_from_saved(dout, saved_tensors, *extras):
        x, alpha, gamma, beta_saved = saved_tensors

        dout = dout.contiguous()

        if beta_saved.numel() == 0 and gamma.numel() != 0:
            beta_for_liger = None
        else:
            beta_for_liger = beta_saved

        dx, dalpha, dgamma, dbeta = _liger_dyt_bwd(
            dout,
            x,
            alpha,
            gamma,
            beta_for_liger,
        )

        dalpha = dalpha.view_as(alpha)

        return dx, dalpha, dgamma, dbeta

    return forward_with_saved, backward_from_saved
