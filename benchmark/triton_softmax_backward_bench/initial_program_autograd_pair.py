import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START

@triton.jit
def _softmax_fwd_kernel(
    X,
    Y,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N_COLS

    x = tl.load(
        X + row * N_COLS + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    m = tl.max(x, axis=0)
    z = x - m
    num = tl.exp(z)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(Y + row * N_COLS + offs, y, mask=mask)


@triton.jit
def _softmax_bwd_kernel(
    DY,
    Y,
    DX,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N_COLS

    dy = tl.load(
        DY + row * N_COLS + offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    y = tl.load(
        Y + row * N_COLS + offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    s = tl.sum(dy * y, axis=0)
    dx = y * (dy - s)

    tl.store(DX + row * N_COLS + offs, dx, mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _num_warps_for_block(block_size: int) -> int:
    if block_size <= 64:
        return 4
    if block_size <= 2048:
        return 4
    if block_size <= 4096:
        return 8
    return 16


def _check_supported_input(x, name: str):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if x.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if x.shape[-1] <= 0:
        raise ValueError(f"{name}.shape[-1] must be positive")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{name} dtype must be float16, bfloat16, or float32")


def _softmax_launch_config(n_cols: int):
    block_size = _next_power_of_2(n_cols)
    if block_size > 131072:
        raise ValueError("last dimension is too large for this Triton softmax kernel")
    num_warps = _num_warps_for_block(block_size)
    return block_size, num_warps


def _softmax_forward_launch(x):
    _check_supported_input(x, "x")

    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols
    y = torch.empty_like(x)

    if x.numel() == 0:
        return y

    block_size, num_warps = _softmax_launch_config(n_cols)
    _softmax_fwd_kernel[(n_rows,)](
        x,
        y,
        N_COLS=n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return y


def _softmax_backward_launch(dy, y):
    _check_supported_input(dy, "dy")
    _check_supported_input(y, "saved y")

    if dy.shape != y.shape:
        raise ValueError("dy and saved y must have the same shape")
    if dy.dtype != y.dtype:
        raise TypeError("dy and saved y must have the same dtype")

    n_cols = dy.shape[-1]
    n_rows = dy.numel() // n_cols
    dx = torch.empty_like(dy)

    if dy.numel() == 0:
        return dx

    block_size, num_warps = _softmax_launch_config(n_cols)
    _softmax_bwd_kernel[(n_rows,)](
        dy,
        y,
        dx,
        N_COLS=n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return dx

# EVOLVE-BLOCK-END


def softmax_forward_with_saved(x):
    y = _softmax_forward_launch(x)
    saved_tensors = (y,)
    return y, saved_tensors


def softmax_backward_from_saved(dy, saved_tensors):
    if not isinstance(saved_tensors, (tuple, list)):
        raise TypeError("saved_tensors must be a tuple or list of tensors")
    if len(saved_tensors) != 1:
        raise ValueError("saved_tensors must contain exactly one tensor")
    (y,) = saved_tensors
    dx = _softmax_backward_launch(dy, y)
    return dx
