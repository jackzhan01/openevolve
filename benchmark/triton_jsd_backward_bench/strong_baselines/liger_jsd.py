import importlib as _importlib
import inspect as _inspect

import torch as _torch

try:
    import torch.distributed.tensor as _torch_distributed_tensor  # noqa: F401
except Exception:
    pass


__all__ = ("liger_available", "make_liger_jsd_autograd_pair_fns")


_LIGER_JSD_MODULE_CANDIDATES = (
    "liger_kernel.ops.jsd",
    "liger_kernel.ops.jsd_loss",
)


def _import_liger_jsd_ops():
    last_exc = None
    for module_name in _LIGER_JSD_MODULE_CANDIDATES:
        try:
            module = _importlib.import_module(module_name)
            jsd_forward = getattr(module, "jsd_forward")
            jsd_backward = getattr(module, "jsd_backward")
            return jsd_forward, jsd_backward
        except Exception as exc:
            last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise ImportError("Could not import Liger JSD raw ops")


def liger_available() -> bool:
    try:
        _import_liger_jsd_ops()
        return True
    except Exception:
        return False


def _call_jsd_forward(jsd_forward, log_q, target):
    return jsd_forward(
        log_q,
        target,
        None,
        0.5,
        -100,
        False,
    )


def _call_jsd_backward(jsd_backward, dlog_q_precomputed, dout):
    try:
        params = _inspect.signature(jsd_backward).parameters
    except (TypeError, ValueError):
        params = {}

    if "in_place" in params:
        return jsd_backward(dlog_q_precomputed, dout, in_place=False)
    if "inplace" in params:
        return jsd_backward(dlog_q_precomputed, dout, inplace=False)

    try:
        return jsd_backward(dlog_q_precomputed, dout, False)
    except TypeError:
        return jsd_backward(dlog_q_precomputed, dout)


def make_liger_jsd_autograd_pair_fns():
    _jsd_forward, _jsd_backward = _import_liger_jsd_ops()

    def forward_with_saved(log_q, target):
        log_q = log_q.contiguous()
        target = target.contiguous()

        original_dtype_marker = log_q.new_empty(())

        compute_log_q = log_q
        compute_target = target
        if log_q.dtype in (_torch.float16, _torch.bfloat16) or target.dtype in (
            _torch.float16,
            _torch.bfloat16,
        ):
            compute_log_q = log_q.float()
            compute_target = target.float()

        result = _call_jsd_forward(_jsd_forward, compute_log_q, compute_target)
        loss, dlog_q_precomputed = result[0], result[1]

        return loss.float(), (dlog_q_precomputed.contiguous(), original_dtype_marker)

    def backward_from_saved(dout, saved_tensors, *extras):
        dlog_q_precomputed, original_dtype_marker = saved_tensors

        # FAIRNESS: no clones here. Liger's jsd_backward is NOT in-place (it returns
        # `grad_output * dX`, a fresh tensor, whenever grad_output != 1.0 — and the harness
        # cotangent is never exactly 1.0), so cloning both tensors only slowed the timed
        # baseline backward and inflated the measured speedups.
        dout_arg = dout
        if dout_arg.dtype != dlog_q_precomputed.dtype:
            dout_arg = dout_arg.to(dtype=dlog_q_precomputed.dtype)

        dlog_q = _call_jsd_backward(_jsd_backward, dlog_q_precomputed, dout_arg)
        if isinstance(dlog_q, tuple):
            dlog_q = dlog_q[0]

        if dlog_q.dtype != original_dtype_marker.dtype:
            dlog_q = dlog_q.to(dtype=original_dtype_marker.dtype)

        return dlog_q

    return forward_with_saved, backward_from_saved
