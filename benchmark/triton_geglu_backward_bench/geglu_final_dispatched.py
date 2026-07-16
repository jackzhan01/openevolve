import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START

@triton.jit
def _geglu_forward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if EVEN:
        a = tl.load(a_ptr + offs).to(tl.float32)
        b = tl.load(b_ptr + offs).to(tl.float32)
    else:
        mask = offs < n_elements
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    a2 = a * a
    z = 0.7978845608028654 * (a + 0.044715 * a * a2)
    t = 2.0 / (1.0 + tl.exp(-2.0 * z)) - 1.0
    c = 0.5 * a * (1.0 + t) * b

    if EVEN:
        tl.store(c_ptr + offs, c)
    else:
        tl.store(c_ptr + offs, c, mask=mask)


@triton.jit
def _geglu_backward_kernel(
    dc_ptr,
    a_ptr,
    b_ptr,
    da_ptr,
    db_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if EVEN:
        dc = tl.load(dc_ptr + offs).to(tl.float32)
        a = tl.load(a_ptr + offs).to(tl.float32)
        b = tl.load(b_ptr + offs).to(tl.float32)
    else:
        mask = offs < n_elements
        dc = tl.load(dc_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    a2 = a * a
    z = 0.7978845608028654 * (a + 0.044715 * a * a2)
    t = 2.0 / (1.0 + tl.exp(-2.0 * z)) - 1.0

    one_plus_t = 1.0 + t
    gelu_a = 0.5 * a * one_plus_t
    dgelu_da = 0.5 * one_plus_t + 0.3989422804014327 * a * (1.0 - t * t) * (1.0 + 0.134145 * a2)

    da = dc * b * dgelu_da
    db = dc * gelu_a

    if EVEN:
        tl.store(da_ptr + offs, da)
        tl.store(db_ptr + offs, db)
    else:
        tl.store(da_ptr + offs, da, mask=mask)
        tl.store(db_ptr + offs, db, mask=mask)


def _launch_geglu_forward(a, b, c):
    n_elements = a.numel()
    if n_elements == 0:
        return
    if n_elements % 1024 == 0:
        block_size = 1024
    elif n_elements % 512 == 0:
        block_size = 512
    elif n_elements % 256 == 0:
        block_size = 256
    elif n_elements % 128 == 0:
        block_size = 128
    else:
        block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _geglu_forward_kernel[grid](
        a,
        b,
        c,
        n_elements,
        BLOCK_SIZE=block_size,
        EVEN=(n_elements % block_size == 0),
        num_warps=4 if block_size >= 512 else 1,
    )


def _launch_geglu_backward(dc, a, b, da, db):
    n_elements = a.numel()
    if n_elements == 0:
        return
    if n_elements % 1024 == 0:
        block_size = 1024
    elif n_elements % 512 == 0:
        block_size = 512
    elif n_elements % 256 == 0:
        block_size = 256
    elif n_elements % 128 == 0:
        block_size = 128
    else:
        block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _geglu_backward_kernel[grid](
        dc,
        a,
        b,
        da,
        db,
        n_elements,
        BLOCK_SIZE=block_size,
        EVEN=(n_elements % block_size == 0),
        num_warps=4 if block_size >= 512 else 1,
    )

# EVOLVE-BLOCK-END


def geglu_forward_with_saved(a, b):
    if a.shape != b.shape:
        raise ValueError("geglu_forward_with_saved expects a and b to have the same shape")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise ValueError("geglu_forward_with_saved expects CUDA tensors")
    if a.dtype != b.dtype:
        raise ValueError("geglu_forward_with_saved expects a and b to have the same dtype")

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    c = torch.empty_like(a)

    _launch_geglu_forward(a, b, c)

    saved_tensors = (a, b)
    return c, saved_tensors


def geglu_backward_from_saved(dc, saved_tensors):
    a, b = saved_tensors

    if dc.shape != a.shape:
        raise ValueError("geglu_backward_from_saved expects dc to have the same shape as saved a")
    if b.shape != a.shape:
        raise ValueError("saved tensors have inconsistent shapes")
    if dc.device.type != "cuda":
        raise ValueError("geglu_backward_from_saved expects CUDA tensors")

    if not dc.is_contiguous():
        dc = dc.contiguous()
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()


    da = torch.empty_like(a)
    db = torch.empty_like(b)

    _launch_geglu_backward(dc, a, b, da, db)

    return da, db

# NOTE: regime collapsed to a single program; this is the small_r1 specialist.
