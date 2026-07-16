import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START

@triton.jit
def _ce_forward_rows_kernel(
    logits_ptr,
    target_ptr,
    lse_ptr,
    loss_vec_ptr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_V)
    mask = offs < V

    x = tl.load(
        logits_ptr + row * V + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    m = tl.max(x, axis=0)
    ex = tl.exp(x - m)
    s = tl.sum(ex, axis=0)
    lse = m + tl.log(s)

    tgt = tl.load(target_ptr + row)
    x_tgt = tl.load(logits_ptr + row * V + tgt).to(tl.float32)

    loss_i = lse - x_tgt

    tl.store(lse_ptr + row, lse)
    tl.store(loss_vec_ptr + row, loss_i)


@triton.jit
def _ce_reduce_loss_kernel(
    loss_vec_ptr,
    loss_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    vals = tl.load(loss_vec_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(vals, axis=0)
    tl.store(loss_ptr, total / N)


@triton.jit
def _ce_backward_kernel(
    dloss_ptr,
    logits_ptr,
    target_ptr,
    lse_ptr,
    dlogits_ptr,
    N: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_V)
    mask = offs < V

    x = tl.load(
        logits_ptr + row * V + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    lse = tl.load(lse_ptr + row).to(tl.float32)
    tgt = tl.load(target_ptr + row)

    p = tl.exp(x - lse)
    onehot = (offs == tgt).to(tl.float32)

    scale = tl.load(dloss_ptr).to(tl.float32) / N
    grad = (p - onehot) * scale

    tl.store(dlogits_ptr + row * V + offs, grad, mask=mask)


def _next_power_of_2_int(x):
    return 1 << (int(x) - 1).bit_length()


def _num_warps_for_block(block):
    block = int(block)
    if block >= 2048:
        return 8
    if block >= 512:
        return 4
    return 1


def _launch_ce_forward(logits, target, lse, loss_vec, loss, N, V):
    block_v = _next_power_of_2_int(V)
    block_n = _next_power_of_2_int(N)

    _ce_forward_rows_kernel[(N,)](
        logits,
        target,
        lse,
        loss_vec,
        V,
        block_v,
        num_warps=_num_warps_for_block(block_v),
    )

    _ce_reduce_loss_kernel[(1,)](
        loss_vec,
        loss,
        N,
        block_n,
        num_warps=_num_warps_for_block(block_n),
    )


def _launch_ce_backward(dloss, logits, target, lse, dlogits, N, V):
    block_v = _next_power_of_2_int(V)

    _ce_backward_kernel[(N,)](
        dloss,
        logits,
        target,
        lse,
        dlogits,
        N,
        V,
        block_v,
        num_warps=_num_warps_for_block(block_v),
    )


# EVOLVE-BLOCK-END


def cross_entropy_forward_with_saved(logits, target):
    if not logits.is_cuda or not target.is_cuda:
        raise ValueError("cross_entropy_forward_with_saved expects CUDA tensors")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [BT, V]")
    if target.ndim != 1:
        raise ValueError("target must have shape [BT]")
    if not logits.is_contiguous():
        raise ValueError("logits must be contiguous")
    if not target.is_contiguous():
        raise ValueError("target must be contiguous")

    n, v = logits.shape
    if target.shape[0] != n:
        raise ValueError("target length must match logits.shape[0]")
    if n <= 0 or v <= 0:
        raise ValueError("logits dimensions must be non-empty")

    lse = torch.empty((n,), device=logits.device, dtype=torch.float32)
    loss_vec = torch.empty((n,), device=logits.device, dtype=torch.float32)
    loss = torch.empty((), device=logits.device, dtype=logits.dtype)

    _launch_ce_forward(logits, target, lse, loss_vec, loss, n, v)

    saved_tensors = (logits, target, lse)
    return loss, saved_tensors


def cross_entropy_backward_from_saved(dloss, saved_tensors):
    logits, target, lse = saved_tensors

    if not dloss.is_cuda or not logits.is_cuda or not target.is_cuda or not lse.is_cuda:
        raise ValueError("cross_entropy_backward_from_saved expects CUDA tensors")
    if logits.ndim != 2:
        raise ValueError("saved logits must have shape [BT, V]")
    if target.ndim != 1:
        raise ValueError("saved target must have shape [BT]")
    if lse.ndim != 1:
        raise ValueError("saved lse must have shape [BT]")
    if not logits.is_contiguous():
        raise ValueError("saved logits must be contiguous")
    if not target.is_contiguous():
        raise ValueError("saved target must be contiguous")
    if not lse.is_contiguous():
        raise ValueError("saved lse must be contiguous")
    if dloss.ndim != 0:
        raise ValueError("dloss must be a scalar tensor")

    n, v = logits.shape
    if target.shape[0] != n or lse.shape[0] != n:
        raise ValueError("saved tensor shapes are inconsistent")

    dlogits = torch.empty_like(logits)

    _launch_ce_backward(dloss, logits, target, lse, dlogits, n, v)

    return dlogits
