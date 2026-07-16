import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START

@triton.jit
def _kl_div_fwd_partials_kernel(
    y_pred_ptr,
    y_true_ptr,
    partials_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    yp = tl.load(y_pred_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    yt = tl.load(y_true_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    yt_safe = tl.where(yt > 0.0, yt, 1.0)
    term = tl.where((yt > 0.0) & mask, yt * (tl.log(yt_safe) - yp), 0.0)
    acc = tl.sum(term, axis=0)

    tl.store(partials_ptr + pid, acc)


@triton.jit
def _sum_partials_kernel(
    in_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    vals = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(vals, axis=0)

    tl.store(out_ptr + pid, acc)


@triton.jit
def _kl_div_fwd_finalize_kernel(
    partials_ptr,
    loss_ptr,
    NUM_PARTIALS: tl.constexpr,
    BT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < NUM_PARTIALS

    vals = tl.load(partials_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(vals, axis=0)
    loss = total / BT

    tl.store(loss_ptr, loss)


@triton.jit
def _kl_div_bwd_kernel(
    dloss_ptr,
    y_true_ptr,
    d_input_ptr,
    N,
    BT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    dloss = tl.load(dloss_ptr).to(tl.float32)
    scale = -dloss / BT

    yt = tl.load(y_true_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = scale * yt

    tl.store(d_input_ptr + offs, out, mask=mask)


def _next_power_of_2_int(x):
    return 1 << (int(x) - 1).bit_length()


def _num_warps_for_block(block):
    block = int(block)
    if block >= 2048:
        return 8
    if block >= 512:
        return 4
    if block >= 128:
        return 2
    return 1


def _launch_kl_div_forward(y_pred, y_true):
    BT, V = y_pred.shape
    N = BT * V

    loss = torch.empty((), device=y_pred.device, dtype=y_pred.dtype)

    block = 1024
    num_partials = triton.cdiv(N, block)
    partials = torch.empty((num_partials,), device=y_pred.device, dtype=torch.float32)

    _kl_div_fwd_partials_kernel[(num_partials,)](
        y_pred,
        y_true,
        partials,
        N,
        BLOCK=block,
        num_warps=4,
    )

    reduce_count = num_partials
    reduce_buf = partials
    reduce_block = 1024

    while reduce_count > reduce_block:
        next_count = triton.cdiv(reduce_count, reduce_block)
        next_buf = torch.empty((next_count,), device=y_pred.device, dtype=torch.float32)
        _sum_partials_kernel[(next_count,)](
            reduce_buf,
            next_buf,
            reduce_count,
            BLOCK=reduce_block,
            num_warps=4,
        )
        reduce_buf = next_buf
        reduce_count = next_count

    final_block = _next_power_of_2_int(reduce_count)
    _kl_div_fwd_finalize_kernel[(1,)](
        reduce_buf,
        loss,
        NUM_PARTIALS=reduce_count,
        BT=BT,
        BLOCK=final_block,
        num_warps=_num_warps_for_block(final_block),
    )

    return loss


def _launch_kl_div_backward(dloss, y_true, out_dtype):
    BT, V = y_true.shape
    N = BT * V

    d_input = torch.empty(y_true.shape, device=y_true.device, dtype=out_dtype)

    block = 1024
    grid = (triton.cdiv(N, block),)

    _kl_div_bwd_kernel[grid](
        dloss,
        y_true,
        d_input,
        N,
        BT=BT,
        BLOCK=block,
        num_warps=4,
    )

    return d_input

# EVOLVE-BLOCK-END


def kl_div_forward_with_saved(y_pred, y_true):
    if not y_pred.is_cuda or not y_true.is_cuda:
        raise ValueError("kl_div_forward_with_saved requires CUDA tensors")
    if y_pred.ndim != 2 or y_true.ndim != 2:
        raise ValueError("y_pred and y_true must both have shape [BT, V]")
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape")
    if y_pred.device != y_true.device:
        raise ValueError("y_pred and y_true must be on the same device")
    if y_pred.shape[0] <= 0 or y_pred.shape[1] <= 0:
        raise ValueError("BT and V must be positive")

    y_pred_c = y_pred.contiguous()
    y_true_c = y_true.contiguous()

    loss = _launch_kl_div_forward(y_pred_c, y_true_c)

    dtype_anchor = torch.empty((0,), device=y_pred.device, dtype=y_pred.dtype)
    saved_tensors = (y_true_c, dtype_anchor)
    return loss, saved_tensors


def kl_div_backward_from_saved(dloss, saved_tensors):
    if torch.is_tensor(saved_tensors):
        y_true = saved_tensors
        dtype_anchor = saved_tensors
    else:
        y_true = saved_tensors[0]
        dtype_anchor = saved_tensors[1] if len(saved_tensors) > 1 else saved_tensors[0]

    if not y_true.is_cuda:
        raise ValueError("saved y_true must be a CUDA tensor")
    if y_true.ndim != 2:
        raise ValueError("saved y_true must have shape [BT, V]")
    if y_true.shape[0] <= 0 or y_true.shape[1] <= 0:
        raise ValueError("BT and V must be positive")

    y_true_c = y_true.contiguous()

    if not torch.is_tensor(dloss):
        dloss_t = torch.tensor(dloss, device=y_true_c.device, dtype=torch.float32)
    else:
        if dloss.numel() != 1:
            raise ValueError("dloss must be a scalar cotangent")
        if dloss.device != y_true_c.device:
            dloss_t = dloss.to(device=y_true_c.device)
        else:
            dloss_t = dloss
        dloss_t = dloss_t.contiguous()

    return _launch_kl_div_backward(dloss_t, y_true_c, dtype_anchor.dtype)
