import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START

_GELU_K_VALUE = 0.7978845608028654
_GELU_C_VALUE = 0.044715
_GELU_3C_VALUE = 0.134145


@triton.jit
def _tanh_compat(x):
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


@triton.jit
def _geglu_forward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    tanh_ptr,
    a_s0,
    a_s1,
    b_s0,
    b_s1,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    GELU_K: tl.constexpr,
    GELU_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < ROWS) & (offs_n[None, :] < COLS)

    in_a_offsets = offs_m[:, None] * a_s0 + offs_n[None, :] * a_s1
    in_b_offsets = offs_m[:, None] * b_s0 + offs_n[None, :] * b_s1
    out_offsets = offs_m[:, None] * COLS + offs_n[None, :]

    a = tl.load(a_ptr + in_a_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + in_b_offsets, mask=mask, other=0.0).to(tl.float32)

    a2 = a * a
    z = GELU_K * (a + GELU_C * a2 * a)
    t = _tanh_compat(z)

    gelu_a = 0.5 * a * (1.0 + t)
    c = gelu_a * b

    tl.store(c_ptr + out_offsets, c, mask=mask)
    tl.store(tanh_ptr + out_offsets, t, mask=mask)


@triton.jit
def _geglu_backward_kernel(
    dc_ptr,
    a_ptr,
    b_ptr,
    tanh_ptr,
    da_ptr,
    db_ptr,
    dc_s0,
    dc_s1,
    a_s0,
    a_s1,
    b_s0,
    b_s1,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    GELU_K: tl.constexpr,
    GELU_3C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < ROWS) & (offs_n[None, :] < COLS)

    dc_offsets = offs_m[:, None] * dc_s0 + offs_n[None, :] * dc_s1
    a_offsets = offs_m[:, None] * a_s0 + offs_n[None, :] * a_s1
    b_offsets = offs_m[:, None] * b_s0 + offs_n[None, :] * b_s1
    out_offsets = offs_m[:, None] * COLS + offs_n[None, :]

    dc = tl.load(dc_ptr + dc_offsets, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + a_offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + b_offsets, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(tanh_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)

    a2 = a * a

    one_plus_t = 1.0 + t
    gelu_a = 0.5 * a * one_plus_t

    one_minus_t2 = 1.0 - t * t
    dz_da = GELU_K * (1.0 + GELU_3C * a2)
    dgelu_da = 0.5 * one_plus_t + 0.5 * a * one_minus_t2 * dz_da

    db = dc * gelu_a
    da = dc * b * dgelu_da

    tl.store(da_ptr + out_offsets, da, mask=mask)
    tl.store(db_ptr + out_offsets, db, mask=mask)


def _select_geglu_block(rows, cols):
    if cols <= 64:
        return 8, 64
    if cols <= 128:
        return 4, 128
    if cols <= 256:
        return 2, 256
    return 1, 1024


def _launch_geglu_forward(a, b):
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("geglu_forward_with_saved expects CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("geglu_forward_with_saved expects rank-2 tensors")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if a.device != b.device:
        raise ValueError("a and b must be on the same device")

    rows = a.shape[0]
    cols = a.shape[1]

    out_dtype = torch.promote_types(a.dtype, b.dtype)
    c = torch.empty((rows, cols), device=a.device, dtype=out_dtype)
    tanh_z = torch.empty((rows, cols), device=a.device, dtype=torch.float32)

    block_m, block_n = _select_geglu_block(rows, cols)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))

    _geglu_forward_kernel[grid](
        a,
        b,
        c,
        tanh_z,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        ROWS=rows,
        COLS=cols,
        GELU_K=_GELU_K_VALUE,
        GELU_C=_GELU_C_VALUE,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )

    return c, (a, b, tanh_z)


def _launch_geglu_backward(dc, saved_tensors):
    if not isinstance(saved_tensors, (tuple, list)) or len(saved_tensors) != 3:
        raise ValueError("saved_tensors must be the tuple returned by geglu_forward_with_saved")

    a, b, tanh_z = saved_tensors

    if not dc.is_cuda or not a.is_cuda or not b.is_cuda or not tanh_z.is_cuda:
        raise ValueError("geglu_backward_from_saved expects CUDA tensors")
    if a.ndim != 2 or b.ndim != 2 or dc.ndim != 2 or tanh_z.ndim != 2:
        raise ValueError("geglu_backward_from_saved expects rank-2 tensors")
    if dc.shape != a.shape or b.shape != a.shape or tanh_z.shape != a.shape:
        raise ValueError("dc and saved tensors must have matching shapes")
    if dc.device != a.device or b.device != a.device or tanh_z.device != a.device:
        raise ValueError("dc and saved tensors must be on the same device")

    rows = a.shape[0]
    cols = a.shape[1]

    da = torch.empty((rows, cols), device=a.device, dtype=a.dtype)
    db = torch.empty((rows, cols), device=b.device, dtype=b.dtype)

    block_m, block_n = _select_geglu_block(rows, cols)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))

    _geglu_backward_kernel[grid](
        dc,
        a,
        b,
        tanh_z,
        da,
        db,
        dc.stride(0),
        dc.stride(1),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        ROWS=rows,
        COLS=cols,
        GELU_K=_GELU_K_VALUE,
        GELU_3C=_GELU_3C_VALUE,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )

    return da, db


# EVOLVE-BLOCK-END


def geglu_forward_with_saved(a, b):
    return _launch_geglu_forward(a, b)


def geglu_backward_from_saved(dc, saved_tensors):
    return _launch_geglu_backward(dc, saved_tensors)
