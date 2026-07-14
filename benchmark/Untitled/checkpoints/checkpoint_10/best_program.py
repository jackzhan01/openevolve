import torch
import triton
import triton.language as tl


# EVOLVE-BLOCK-START

@triton.jit
def _layernorm_fwd_kernel(
    X,
    W,
    B,
    Y,
    XHAT,
    RSTD,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)

    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.rsqrt(var + EPS)
    xhat = xc * rstd

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xhat * w + b

    tl.store(Y + row * N + cols, y, mask=mask)
    tl.store(XHAT + row * N + cols, xhat, mask=mask)
    tl.store(RSTD + row, rstd)





@triton.jit
def _layernorm_bwd_tile_kernel(
    DY, XHAT, RSTD, W, DX, PDW, PDB,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offs = rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)

    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    xh = tl.load(XHAT + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + rows, mask=rows < M, other=0.0).to(tl.float32)

    g = tl.where(mask, dy * w[None, :], 0.0)
    xh = tl.where(mask, xh, 0.0)
    dy = tl.where(mask, dy, 0.0)

    sg = tl.sum(g, axis=1) / N
    sgxh = tl.sum(g * xh, axis=1) / N
    dx = rstd[:, None] * (g - sg[:, None] - xh * sgxh[:, None])
    tl.store(DX + offs, dx, mask=mask)

    tl.store(PDW + pid * N + cols, tl.sum(dy * xh, axis=0), mask=cols < N)
    tl.store(PDB + pid * N + cols, tl.sum(dy, axis=0), mask=cols < N)


@triton.jit
def _layernorm_bwd_final_kernel(
    PDW, PDB, DW, DB,
    R: tl.constexpr, N: tl.constexpr,
    BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr,
):
    r = tl.arange(0, BLOCK_R)
    c = tl.arange(0, BLOCK_N)
    mask = (r[:, None] < R) & (c[None, :] < N)
    offs = r[:, None] * N + c[None, :]
    dw = tl.sum(tl.load(PDW + offs, mask=mask, other=0.0).to(tl.float32), axis=0)
    db = tl.sum(tl.load(PDB + offs, mask=mask, other=0.0).to(tl.float32), axis=0)
    tl.store(DW + c, dw, mask=c < N)
    tl.store(DB + c, db, mask=c < N)


@triton.jit
def _layernorm_bwd_dx_kernel(
    DY, XHAT, RSTD, W, DX,
    M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK_N)
    mask = c < N

    offs = row * N + c
    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    xh = tl.load(XHAT + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + c, mask=mask, other=0.0).to(tl.float32)

    dy = tl.where(mask, dy, 0.0)
    xh = tl.where(mask, xh, 0.0)
    g = tl.where(mask, dy * w, 0.0)

    sg = tl.sum(g, axis=0) / N
    sgxh = tl.sum(g * xh, axis=0) / N
    rstd = tl.load(RSTD + row).to(tl.float32)

    dx = rstd * (g - sg - xh * sgxh)
    tl.store(DX + offs, dx, mask=mask)


@triton.jit
def _layernorm_bwd_dwdb_cols_kernel(
    DY, XHAT, DW, DB,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    r = tl.arange(0, BLOCK_M)
    c = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    offs = r[:, None] * N + c[None, :]
    mask = (r[:, None] < M) & (c[None, :] < N)

    dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
    xh = tl.load(XHAT + offs, mask=mask, other=0.0).to(tl.float32)
    dy = tl.where(mask, dy, 0.0)
    xh = tl.where(mask, xh, 0.0)

    tl.store(DW + c, tl.sum(dy * xh, axis=0), mask=c < N)
    tl.store(DB + c, tl.sum(dy, axis=0), mask=c < N)


def _layernorm_forward_triton(x, weight, bias, eps):
    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()

    M, N = x_c.shape
    y = torch.empty_like(x_c)
    xhat = torch.empty((M, N), device=x_c.device, dtype=(torch.float32 if x_c.dtype == torch.float32 else x_c.dtype))
    rstd = torch.empty((M,), device=x_c.device, dtype=torch.float32)

    block_n = int(triton.next_power_of_2(N))

    _layernorm_fwd_kernel[(M,)](
        x_c,
        w_c,
        b_c,
        y,
        xhat,
        rstd,
        M,
        N,
        float(eps),
        BLOCK_N=block_n,
        num_warps=(4 if block_n >= 2048 else (2 if block_n >= 1024 else 1)),
    )

    x_dtype_marker = torch.empty((0,), device=x_c.device, dtype=x.dtype)
    bias_dtype_marker = torch.empty((0,), device=b_c.device, dtype=bias.dtype)
    saved_tensors = (xhat, rstd, w_c, x_dtype_marker, bias_dtype_marker)
    return y, saved_tensors


def _layernorm_backward_triton(dy, saved_tensors):
    xhat, rstd, weight, x_dtype_marker, bias_dtype_marker = saved_tensors

    dy_c = dy.contiguous()
    M, N = dy_c.shape

    dx = torch.empty((M, N), device=dy_c.device, dtype=x_dtype_marker.dtype)
    dweight = torch.empty_like(weight)
    dbias = torch.empty((N,), device=dy_c.device, dtype=bias_dtype_marker.dtype)

    block_n = int(triton.next_power_of_2(N))

    if M >= 16 and block_n >= 2048:
        _layernorm_bwd_dx_kernel[(M,)](
            dy_c,
            xhat,
            rstd,
            weight,
            dx,
            M,
            N,
            BLOCK_N=block_n,
            num_warps=(4 if block_n >= 2048 else 1),
        )
        dwdb_bn = 256
        _layernorm_bwd_dwdb_cols_kernel[(triton.cdiv(N, dwdb_bn),)](
            dy_c,
            xhat,
            dweight,
            dbias,
            M,
            N,
            BLOCK_M=int(triton.next_power_of_2(M)),
            BLOCK_N=dwdb_bn,
            num_warps=4,
        )
        return dx, dweight, dbias

    tile_m = 8 if block_n >= 2048 else 16
    while tile_m > 1 and tile_m * block_n > 32768:
        tile_m //= 2
    tile_m = min(tile_m, int(triton.next_power_of_2(M)))
    ntiles = triton.cdiv(M, tile_m)

    if ntiles == 1:
        pdw, pdb = dweight, dbias
    else:
        pdw = torch.empty((ntiles, N), device=dy_c.device, dtype=torch.float32)
        pdb = torch.empty((ntiles, N), device=dy_c.device, dtype=torch.float32)

    _layernorm_bwd_tile_kernel[(ntiles,)](
        dy_c,
        xhat,
        rstd,
        weight,
        dx,
        pdw,
        pdb,
        M,
        N,
        BLOCK_M=tile_m,
        BLOCK_N=block_n,
        num_warps=(8 if block_n >= 4096 else (4 if block_n >= 1024 else 1)),
    )

    if ntiles != 1:
        block_r = int(triton.next_power_of_2(ntiles))
        _layernorm_bwd_final_kernel[(1,)](
            pdw,
            pdb,
            dweight,
            dbias,
            ntiles,
            N,
            BLOCK_R=block_r,
            BLOCK_N=block_n,
            num_warps=(4 if block_n >= 1024 else 1),
        )

    return dx, dweight, dbias


# EVOLVE-BLOCK-END


def _supports_triton_layernorm(x, weight, bias):
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if not (x.is_cuda and weight.is_cuda and bias.is_cuda):
        return False
    if x.ndim != 2 or weight.ndim != 1 or bias.ndim != 1:
        return False
    if x.shape[1] != weight.shape[0] or weight.shape[0] != bias.shape[0]:
        return False
    if x.dtype not in supported_dtypes:
        return False
    if weight.dtype not in supported_dtypes:
        return False
    if bias.dtype not in supported_dtypes:
        return False

    M, N = x.shape
    block_m = int(triton.next_power_of_2(M))
    block_n = int(triton.next_power_of_2(N))
    if block_n > 131072:
        return False
    if block_m * block_n > 131072:
        return False
    return True


def _layernorm_forward_torch_math(x, weight, bias, eps):
    if x.ndim != 2:
        raise ValueError("layernorm_forward_with_saved expects a 2D input tensor")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("weight and bias must be 1D tensors")
    if x.shape[1] != weight.shape[0] or weight.shape[0] != bias.shape[0]:
        raise ValueError("incompatible x, weight, and bias shapes")

    with torch.no_grad():
        x_c = x.contiguous()
        w_c = weight.contiguous()
        b_c = bias.contiguous()

        acc_dtype = torch.float32
        if x_c.dtype == torch.float64 or w_c.dtype == torch.float64 or b_c.dtype == torch.float64:
            acc_dtype = torch.float64

        x_f = x_c.to(acc_dtype)
        mean = x_f.mean(dim=1, keepdim=True)
        xc = x_f - mean
        var = (xc * xc).mean(dim=1, keepdim=True)
        rstd_2d = torch.rsqrt(var + float(eps))
        xhat = xc * rstd_2d

        y = xhat * w_c.to(acc_dtype).view(1, -1) + b_c.to(acc_dtype).view(1, -1)
        y = y.to(x_c.dtype)

        rstd = rstd_2d.reshape(-1).to(torch.float32 if acc_dtype == torch.float32 else acc_dtype)
        xhat_saved = xhat.to(torch.float32 if acc_dtype == torch.float32 else acc_dtype)
        x_dtype_marker = torch.empty((0,), device=x_c.device, dtype=x.dtype)
        bias_dtype_marker = torch.empty((0,), device=b_c.device, dtype=bias.dtype)

    return y, (xhat_saved, rstd, w_c, x_dtype_marker, bias_dtype_marker)


def _layernorm_backward_torch_math(dy, saved_tensors):
    xhat, rstd, weight, x_dtype_marker, bias_dtype_marker = saved_tensors

    with torch.no_grad():
        dy_c = dy.contiguous()
        M, N = dy_c.shape

        acc_dtype = torch.float32
        if dy_c.dtype == torch.float64 or xhat.dtype == torch.float64 or weight.dtype == torch.float64:
            acc_dtype = torch.float64

        dy_f = dy_c.to(acc_dtype)
        xhat_f = xhat.to(acc_dtype)
        w_f = weight.to(acc_dtype).view(1, N)
        rstd_f = rstd.to(acc_dtype).view(M, 1)

        g = dy_f * w_f
        mean_g = g.mean(dim=1, keepdim=True)
        mean_g_xhat = (g * xhat_f).mean(dim=1, keepdim=True)

        dx = rstd_f * (g - mean_g - xhat_f * mean_g_xhat)
        dweight = (dy_f * xhat_f).sum(dim=0)
        dbias = dy_f.sum(dim=0)

        dx = dx.to(x_dtype_marker.dtype)
        dweight = dweight.to(weight.dtype)
        dbias = dbias.to(bias_dtype_marker.dtype)

    return dx, dweight, dbias


def layernorm_forward_with_saved(x, weight, bias, eps=1e-5):
    if _supports_triton_layernorm(x, weight, bias):
        return _layernorm_forward_triton(x, weight, bias, eps)
    return _layernorm_forward_torch_math(x, weight, bias, eps)


def layernorm_backward_from_saved(dy, saved_tensors, eps=1e-5):
    _ = eps
    if not isinstance(saved_tensors, (tuple, list)):
        raise TypeError("saved_tensors must be a tuple or list of tensors")
    if len(saved_tensors) != 5:
        raise ValueError("saved_tensors must contain (xhat, rstd, weight, x_dtype_marker, bias_dtype_marker)")

    xhat, rstd, weight, x_dtype_marker, bias_dtype_marker = saved_tensors
    if not all(torch.is_tensor(t) for t in (xhat, rstd, weight, x_dtype_marker, bias_dtype_marker)):
        raise TypeError("all saved_tensors entries must be tensors")

    if (
        dy.is_cuda
        and xhat.is_cuda
        and rstd.is_cuda
        and weight.is_cuda
        and dy.ndim == 2
        and xhat.ndim == 2
        and rstd.ndim == 1
        and weight.ndim == 1
        and dy.shape == xhat.shape
        and dy.shape[0] == rstd.shape[0]
        and dy.shape[1] == weight.shape[0]
        and dy.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and xhat.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        M, N = dy.shape
        block_m = int(triton.next_power_of_2(M))
        block_n = int(triton.next_power_of_2(N))
        if block_m * block_n <= 131072:
            return _layernorm_backward_triton(dy, saved_tensors)

    return _layernorm_backward_torch_math(dy, saved_tensors)
