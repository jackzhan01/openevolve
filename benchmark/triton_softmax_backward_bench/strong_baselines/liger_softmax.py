"""Liger softmax strong-baseline wrappers."""

from __future__ import annotations

from typing import Callable as _Callable
import inspect as _inspect

import torch as _torch

try:
    import torch.distributed.tensor as _torch_distributed_tensor  # noqa: F401
except Exception:  # pragma: no cover
    _torch_distributed_tensor = None  # type: ignore[assignment]


__all__ = ["liger_available", "make_liger_softmax_autograd_pair_fns"]


_triton = None
_tl = None
_FIXED_SOFTMAX_MULTI_BLOCK_FORWARD_KERNEL = None
_FIXED_SOFTMAX_MULTI_BLOCK_BACKWARD_KERNEL = None


def liger_available() -> bool:
    try:
        import liger_kernel.ops.softmax as _softmax_mod  # noqa: F401
        import liger_kernel.ops.utils as _utils_mod  # noqa: F401

        return hasattr(_softmax_mod, "_softmax_forward") and hasattr(_softmax_mod, "_softmax_backward")
    except Exception:
        return False


def _fallback_num_warps(_block_size: int) -> int:
    if _block_size >= 32768:
        return 32
    if _block_size >= 8192:
        return 16
    if _block_size >= 2048:
        return 8
    return 4


def _get_fixed_softmax_multi_block_forward_kernel():
    global _triton, _tl, _FIXED_SOFTMAX_MULTI_BLOCK_FORWARD_KERNEL

    if _FIXED_SOFTMAX_MULTI_BLOCK_FORWARD_KERNEL is not None:
        return _FIXED_SOFTMAX_MULTI_BLOCK_FORWARD_KERNEL

    import triton as _triton_import
    import triton.language as _tl_import

    _triton = _triton_import
    _tl = _tl_import

    @_triton.jit
    def _fixed_softmax_multi_block_forward_kernel(
        Y_ptr,
        Y_row_stride,
        X_ptr,
        X_row_stride,
        n_cols,
        BLOCK_SIZE: _tl.constexpr,
    ):
        row_id = _tl.program_id(0)
        offs = _tl.arange(0, BLOCK_SIZE)

        m = -float("inf")
        d = 0.0

        for block_start in range(0, n_cols, BLOCK_SIZE):
            cols = block_start + offs
            mask = cols < n_cols
            x = _tl.load(
                X_ptr + row_id * X_row_stride + cols,
                mask=mask,
                other=-float("inf"),
            ).to(_tl.float32)

            block_m = _tl.max(x, axis=0)
            new_m = _tl.maximum(m, block_m)
            d = d * _tl.exp(m - new_m) + _tl.sum(_tl.exp(x - new_m), axis=0)
            m = new_m

        for block_start in range(0, n_cols, BLOCK_SIZE):
            cols = block_start + offs
            mask = cols < n_cols
            x = _tl.load(
                X_ptr + row_id * X_row_stride + cols,
                mask=mask,
                other=-float("inf"),
            ).to(_tl.float32)
            y = _tl.exp(x - m) / d
            _tl.store(Y_ptr + row_id * Y_row_stride + cols, y, mask=mask)

    _FIXED_SOFTMAX_MULTI_BLOCK_FORWARD_KERNEL = _fixed_softmax_multi_block_forward_kernel
    return _FIXED_SOFTMAX_MULTI_BLOCK_FORWARD_KERNEL


def _get_fixed_softmax_multi_block_backward_kernel():
    global _triton, _tl, _FIXED_SOFTMAX_MULTI_BLOCK_BACKWARD_KERNEL

    if _FIXED_SOFTMAX_MULTI_BLOCK_BACKWARD_KERNEL is not None:
        return _FIXED_SOFTMAX_MULTI_BLOCK_BACKWARD_KERNEL

    import triton as _triton_import
    import triton.language as _tl_import

    _triton = _triton_import
    _tl = _tl_import

    @_triton.jit
    def _fixed_softmax_multi_block_backward_kernel(
        DX_ptr,
        DX_row_stride,
        DY_ptr,
        DY_row_stride,
        Y_ptr,
        Y_row_stride,
        n_cols,
        BLOCK_SIZE: _tl.constexpr,
    ):
        row_id = _tl.program_id(0)
        offs = _tl.arange(0, BLOCK_SIZE)

        s = 0.0
        for block_start in range(0, n_cols, BLOCK_SIZE):
            cols = block_start + offs
            mask = cols < n_cols
            y = _tl.load(Y_ptr + row_id * Y_row_stride + cols, mask=mask, other=0.0).to(_tl.float32)
            dy = _tl.load(DY_ptr + row_id * DY_row_stride + cols, mask=mask, other=0.0).to(_tl.float32)
            s += _tl.sum(dy * y, axis=0)

        for block_start in range(0, n_cols, BLOCK_SIZE):
            cols = block_start + offs
            mask = cols < n_cols
            y = _tl.load(Y_ptr + row_id * Y_row_stride + cols, mask=mask, other=0.0).to(_tl.float32)
            dy = _tl.load(DY_ptr + row_id * DY_row_stride + cols, mask=mask, other=0.0).to(_tl.float32)
            dx = y * (dy - s)
            _tl.store(DX_ptr + row_id * DX_row_stride + cols, dx, mask=mask)

    _FIXED_SOFTMAX_MULTI_BLOCK_BACKWARD_KERNEL = _fixed_softmax_multi_block_backward_kernel
    return _FIXED_SOFTMAX_MULTI_BLOCK_BACKWARD_KERNEL


class _FixedSoftmaxMultiBlockForwardLauncher:
    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            kernel = _get_fixed_softmax_multi_block_forward_kernel()
            kwargs = dict(kwargs)

            block_size = kwargs.pop("BLOCK_SIZE", None)
            if block_size is None:
                block_size = kwargs.pop("block_size", None)
            if block_size is None:
                block_size = kwargs.pop("BLOCK", None)

            args = list(args)
            if block_size is None and len(args) >= 6:
                block_size = args[5]
                args = args[:5]

            if block_size is None:
                raise TypeError("missing BLOCK_SIZE for Liger softmax multi-block forward kernel")
            if len(args) < 5:
                raise TypeError("unexpected Liger softmax multi-block forward kernel argument list")

            return kernel[grid](
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                BLOCK_SIZE=int(block_size),
                **kwargs,
            )

        return _launch


class _FixedSoftmaxMultiBlockBackwardLauncher:
    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            kernel = _get_fixed_softmax_multi_block_backward_kernel()
            kwargs = dict(kwargs)

            block_size = kwargs.pop("BLOCK_SIZE", None)
            if block_size is None:
                block_size = kwargs.pop("block_size", None)
            if block_size is None:
                block_size = kwargs.pop("BLOCK", None)

            args = list(args)
            if block_size is None and len(args) in (6, 8):
                block_size = args[-1]
                args = args[:-1]

            if block_size is None:
                raise TypeError("missing BLOCK_SIZE for Liger softmax multi-block backward kernel")

            if len(args) >= 7:
                dx_ptr, dx_stride, dy_ptr, dy_stride, y_ptr, y_stride, n_cols = args[:7]
            elif len(args) >= 5:
                dy_ptr, dy_stride, y_ptr, y_stride, n_cols = args[:5]
                dx_ptr = dy_ptr
                dx_stride = dy_stride
            else:
                raise TypeError("unexpected Liger softmax multi-block backward kernel argument list")

            return kernel[grid](
                dx_ptr,
                dx_stride,
                dy_ptr,
                dy_stride,
                y_ptr,
                y_stride,
                n_cols,
                BLOCK_SIZE=int(block_size),
                **kwargs,
            )

        return _launch


def _patch_liger_softmax_multiblock_kernels(_softmax_mod, _softmax_forward, _softmax_backward) -> None:
    fwd_launcher = _FixedSoftmaxMultiBlockForwardLauncher()
    bwd_launcher = _FixedSoftmaxMultiBlockBackwardLauncher()

    fwd_globals = getattr(_softmax_forward, "__globals__", {})
    bwd_globals = getattr(_softmax_backward, "__globals__", {})

    if "_softmax_multi_block_forward_kernel" in fwd_globals:
        fwd_globals["_softmax_multi_block_forward_kernel"] = fwd_launcher
    if hasattr(_softmax_mod, "_softmax_multi_block_forward_kernel"):
        setattr(_softmax_mod, "_softmax_multi_block_forward_kernel", fwd_launcher)

    if "_softmax_multi_block_backward_kernel" in bwd_globals:
        bwd_globals["_softmax_multi_block_backward_kernel"] = bwd_launcher
    if hasattr(_softmax_mod, "_softmax_multi_block_backward_kernel"):
        setattr(_softmax_mod, "_softmax_multi_block_backward_kernel", bwd_launcher)


def make_liger_softmax_autograd_pair_fns() -> tuple[_Callable, _Callable]:
    """Return raw-Liger row-wise softmax pair: forward(x)->(y,(y,)), backward(dy,(y,))->dx."""
    import liger_kernel.ops.softmax as _softmax_mod
    import liger_kernel.ops.utils as _utils_mod
    from liger_kernel.ops.softmax import _softmax_backward, _softmax_forward
    from liger_kernel.ops.utils import calculate_settings as _utils_calculate_settings

    _patch_liger_softmax_multiblock_kernels(_softmax_mod, _softmax_forward, _softmax_backward)

    _softmax_forward_globals = getattr(_softmax_forward, "__globals__", {})
    _softmax_backward_globals = getattr(_softmax_backward, "__globals__", {})

    _original_calculate_settings = _softmax_forward_globals.get(
        "calculate_settings",
        getattr(_softmax_mod, "calculate_settings", _utils_calculate_settings),
    )
    _max_fused_size = int(getattr(_utils_mod, "MAX_FUSED_SIZE", 65536))

    def _derive_settings(n_cols: int) -> tuple[int, int]:
        n_cols = int(n_cols)
        try:
            block_size, num_warps = _original_calculate_settings(n_cols)
            return int(block_size), int(num_warps)
        except Exception:
            if n_cols <= _max_fused_size:
                raise

            try:
                block_size, num_warps = _original_calculate_settings(_max_fused_size)
                return int(block_size), int(num_warps)
            except Exception:
                return int(_max_fused_size), _fallback_num_warps(int(_max_fused_size))

    def _patched_calculate_settings(n_cols: int) -> tuple[int, int]:
        return _derive_settings(int(n_cols))

    def _with_patched_calculate_settings(fn):
        sentinel = object()

        old_fwd_global = _softmax_forward_globals.get("calculate_settings", sentinel)
        old_bwd_global = _softmax_backward_globals.get("calculate_settings", sentinel)
        old_module = getattr(_softmax_mod, "calculate_settings", sentinel)

        _softmax_forward_globals["calculate_settings"] = _patched_calculate_settings
        _softmax_backward_globals["calculate_settings"] = _patched_calculate_settings
        setattr(_softmax_mod, "calculate_settings", _patched_calculate_settings)

        try:
            return fn()
        finally:
            if old_fwd_global is sentinel:
                _softmax_forward_globals.pop("calculate_settings", None)
            else:
                _softmax_forward_globals["calculate_settings"] = old_fwd_global

            if old_bwd_global is sentinel:
                _softmax_backward_globals.pop("calculate_settings", None)
            else:
                _softmax_backward_globals["calculate_settings"] = old_bwd_global

            if old_module is sentinel:
                try:
                    delattr(_softmax_mod, "calculate_settings")
                except AttributeError:
                    pass
            else:
                setattr(_softmax_mod, "calculate_settings", old_module)

    try:
        _bwd_sig = _inspect.signature(_softmax_backward)
        _bwd_params = [
            p
            for p in _bwd_sig.parameters.values()
            if p.kind
            in (
                _inspect.Parameter.POSITIONAL_ONLY,
                _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                _inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        _has_var_kwargs = any(
            p.kind == _inspect.Parameter.VAR_KEYWORD for p in _bwd_sig.parameters.values()
        )
    except Exception:
        _bwd_params = []
        _has_var_kwargs = False

    def _call_raw_backward(dout_2d: _torch.Tensor, y_2d: _torch.Tensor) -> _torch.Tensor:
        n_cols = int(y_2d.shape[-1])
        block_size, num_warps = _derive_settings(n_cols)
        multi_block_launch = bool(n_cols > block_size)

        def _call():
            if not _bwd_params or len(_bwd_params) <= 2:
                return _softmax_backward(dout_2d, y_2d)

            kwargs = {}
            unknown_required = False

            for param in _bwd_params[2:]:
                name = param.name
                lname = name.lower()

                if lname in ("block_size", "blocksize") or name == "BLOCK_SIZE":
                    kwargs[name] = block_size
                elif lname == "num_warps":
                    kwargs[name] = num_warps
                elif lname in ("multi_block_launch", "multiblock_launch", "multi_block", "multiblock") or (
                    "multi" in lname and "block" in lname
                ):
                    kwargs[name] = multi_block_launch
                elif lname in ("n_cols", "ncols", "num_cols", "cols"):
                    kwargs[name] = n_cols
                elif lname in ("in_place", "inplace"):
                    kwargs[name] = False
                elif param.default is _inspect.Parameter.empty and not _has_var_kwargs:
                    unknown_required = True

            if not unknown_required:
                return _softmax_backward(dout_2d, y_2d, **kwargs)

            return _softmax_backward(dout_2d, y_2d)

        out = _with_patched_calculate_settings(_call)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def forward_with_saved(x: _torch.Tensor):
        x_contig = x.contiguous()
        original_shape = x_contig.shape

        if x_contig.ndim == 0:
            x_2d = x_contig.reshape(1, 1)
        else:
            x_2d = x_contig.reshape(-1, int(original_shape[-1]))

        out = _with_patched_calculate_settings(lambda: _softmax_forward(x_2d))
        y_2d = out[0] if isinstance(out, tuple) else out
        y = y_2d.reshape(original_shape)

        return y, (y,)

    def backward_from_saved(dout: _torch.Tensor, saved_tensors, *extras):
        (y,) = saved_tensors

        y_contig = y.contiguous().clone()
        dout_contig = dout.contiguous().clone()

        original_shape = y_contig.shape

        if y_contig.ndim == 0:
            y_2d = y_contig.reshape(1, 1)
            dout_2d = dout_contig.reshape(1, 1)
        else:
            n_cols = int(original_shape[-1])
            y_2d = y_contig.reshape(-1, n_cols)
            dout_2d = dout_contig.reshape(-1, n_cols)

        dx_2d = _call_raw_backward(dout_2d, y_2d)
        return dx_2d.reshape(original_shape)

    return forward_with_saved, backward_from_saved
