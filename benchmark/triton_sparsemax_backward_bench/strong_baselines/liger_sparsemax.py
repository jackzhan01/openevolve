import torch as _torch

try:
    import torch.distributed.tensor as _torch_distributed_tensor  # noqa: F401
except Exception:
    _torch_distributed_tensor = None

try:
    from liger_kernel.ops.sparsemax import (
        _sparsemax_backward as _liger_sparsemax_backward,
        _sparsemax_forward as _liger_sparsemax_forward,
    )

    _LIGER_IMPORT_ERROR = None
except Exception as _exc:
    _liger_sparsemax_forward = None
    _liger_sparsemax_backward = None
    _LIGER_IMPORT_ERROR = _exc

__all__ = ("liger_available", "make_liger_sparsemax_autograd_pair_fns")


def liger_available() -> bool:
    return _liger_sparsemax_forward is not None and _liger_sparsemax_backward is not None


def make_liger_sparsemax_autograd_pair_fns() -> tuple:
    if not liger_available():
        raise ImportError("Liger sparsemax raw ops are not available.") from _LIGER_IMPORT_ERROR

    _raw_forward = _liger_sparsemax_forward
    _raw_backward = _liger_sparsemax_backward

    def forward_with_saved(x: _torch.Tensor, dim: int = -1):
        x = x.contiguous()
        y, out_flat = _raw_forward(x, dim)
        return y, (out_flat,)

    def backward_from_saved(dout: _torch.Tensor, saved_tensors: tuple, *extras):
        (out_flat,) = saved_tensors
        dim = extras[0] if extras else -1
        if len(extras) > 1:
            raise TypeError("liger sparsemax backward expects at most one extra argument: dim")
        dx = _raw_backward(dout.contiguous(), out_flat, dim)
        return dx

    return forward_with_saved, backward_from_saved
