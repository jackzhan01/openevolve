"""Naive element-wise Triton backward for SwiGLU.

SwiGLU backward is purely element-wise (no reductions), so this naive
implementation uses a single flat element-wise kernel with BLOCK_SIZE=1024.
It serves as a readable correctness reference and performance lower bound.

    sig_a  = sigmoid(a)
    silu_a = a * sig_a
    db     = dc * silu_a
    da     = dc * b * (silu_a * (1 - sig_a) + sig_a)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_backward_naive_kernel(
    dc_ptr,
    a_ptr,
    b_ptr,
    da_ptr,
    db_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dc = tl.load(dc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig_a = tl.sigmoid(a)
    silu_a = a * sig_a
    da = dc * b * (silu_a * (1.0 - sig_a) + sig_a)
    db = dc * silu_a

    tl.store(da_ptr + offsets, da, mask=mask)
    tl.store(db_ptr + offsets, db, mask=mask)


def swiglu_backward_naive_triton(
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Element-wise SwiGLU backward; single flat kernel, BLOCK_SIZE=1024."""
    dc = dc.contiguous()
    a = a.contiguous()
    b = b.contiguous()

    n_elements = dc.numel()
    da = torch.empty_like(a, dtype=torch.float32)
    db = torch.empty_like(b, dtype=torch.float32)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _swiglu_backward_naive_kernel[grid](
        dc, a, b, da, db,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return da.to(a.dtype), db.to(b.dtype)
