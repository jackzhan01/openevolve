# strong_baselines/liger_relu_squared.py

from __future__ import annotations

import torch as _torch

try:
    import torch.distributed.tensor as _torch_distributed_tensor  # noqa: F401
except Exception:
    _torch_distributed_tensor = None


def liger_available() -> bool:
    try:
        from liger_kernel.ops.relu_squared import relu_squared_backward as _relu_squared_backward  # noqa: F401
        from liger_kernel.ops.relu_squared import relu_squared_forward as _relu_squared_forward  # noqa: F401

        return True
    except Exception:
        return False


def make_liger_relu_squared_autograd_pair_fns():
    from liger_kernel.ops.relu_squared import relu_squared_backward as _relu_squared_backward
    from liger_kernel.ops.relu_squared import relu_squared_forward as _relu_squared_forward

    def forward_with_saved(x):
        x = x.contiguous()
        y = _relu_squared_forward(x)
        saved_tensors = (x,)
        return y, saved_tensors

    def backward_from_saved(dout, saved_tensors, *extras):
        if not isinstance(saved_tensors, tuple):
            raise TypeError("saved_tensors must be a flat tuple of torch.Tensor objects")
        if len(saved_tensors) != 1:
            raise ValueError("saved_tensors for Liger ReLU-squared must contain exactly one tensor")
        x = saved_tensors[0]
        if not isinstance(x, _torch.Tensor):
            raise TypeError("saved_tensors must contain only torch.Tensor objects")

        x = x.contiguous()
        dout = dout.contiguous()
        dx = _relu_squared_backward(x, dout)
        return dx

    return forward_with_saved, backward_from_saved
