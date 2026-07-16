"""Liger Cross Entropy strong-baseline wrappers.

Wraps ``liger_kernel.ops.cross_entropy`` raw ``cross_entropy_forward`` /
``cross_entropy_backward`` (NOT the autograd Function) to match the benchmark's
saved-tensor autograd-pair API for mean-reduced hard-label cross entropy.

The Liger raw forward computes the input gradient in-place into its input tensor
when ``_input.requires_grad`` is true, and the raw backward only applies the scalar
cotangent to that stored gradient. To avoid mutating benchmark inputs and to make
repeated backward calls pure with respect to saved state, forward uses a private
contiguous clone and backward clones the saved gradient tensor before calling the
raw backward.
"""

from __future__ import annotations

from typing import Callable as _Callable

import torch as _torch
import torch.distributed.tensor as _torch_distributed_tensor  # noqa: F401


__all__ = ["liger_available", "make_liger_cross_entropy_autograd_pair_fns"]


_IGNORE_INDEX = -100
_LSE_SQUARE_SCALE = 0.0
_LABEL_SMOOTHING = 0.0
_REDUCTION = "mean"
_SOFTCAP = None
_RETURN_Z_LOSS = False
_RETURN_TOKEN_ACCURACY = False
_RETURN_PREDICTED_TOKENS = False


def liger_available() -> bool:
    try:
        from liger_kernel.ops.cross_entropy import cross_entropy_backward, cross_entropy_forward  # noqa: F401

        return True
    except Exception:
        return False


def make_liger_cross_entropy_autograd_pair_fns() -> tuple[_Callable, _Callable]:
    """Return ``(forward_with_saved, backward_from_saved)`` matching:

        forward_with_saved(logits, target) -> (loss, saved_tensors)
        backward_from_saved(dloss, saved_tensors) -> dlogits

    ``saved_tensors`` is a flat tuple of tensors only. It contains the private
    tensor into which Liger's forward stored the mean-loss gradient for
    ``dloss == 1``.
    """
    from liger_kernel.ops.cross_entropy import cross_entropy_backward, cross_entropy_forward

    def forward_with_saved(logits: _torch.Tensor, target: _torch.Tensor):
        logits_work = logits.detach().clone(memory_format=_torch.contiguous_format).contiguous()
        logits_work.requires_grad_(True)
        target_work = target.contiguous()

        loss, _z_loss, _token_accuracy, _predicted_tokens, logits_grad = cross_entropy_forward(
            logits_work,
            target_work,
            None,
            _IGNORE_INDEX,
            _LSE_SQUARE_SCALE,
            _LABEL_SMOOTHING,
            _REDUCTION,
            _SOFTCAP,
            _RETURN_Z_LOSS,
            _RETURN_TOKEN_ACCURACY,
            _RETURN_PREDICTED_TOKENS,
        )

        return loss, (logits_grad.detach(),)

    def backward_from_saved(dloss: _torch.Tensor, saved_tensors):
        (saved_logits_grad,) = saved_tensors

        # Liger's raw CE backward multiplies/stores into its input tensor for scalar
        # cotangents, so never pass the saved tensor directly: the benchmark reuses
        # saved_tensors across many backward calls.
        logits_grad = saved_logits_grad.clone(memory_format=_torch.contiguous_format).contiguous()
        dloss = dloss.contiguous()

        return cross_entropy_backward(logits_grad, dloss)

    return forward_with_saved, backward_from_saved
