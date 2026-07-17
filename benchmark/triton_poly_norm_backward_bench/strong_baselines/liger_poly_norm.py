import torch as _torch

try:
    import torch.distributed.tensor as _torch_distributed_tensor  # noqa: F401
except Exception:
    _torch_distributed_tensor = None


def _load_liger():
    from liger_kernel.ops.poly_norm import poly_norm_backward, poly_norm_forward

    return poly_norm_forward, poly_norm_backward


def liger_available() -> bool:
    try:
        _load_liger()
        return True
    except Exception:
        return False


def _poly_norm_vector_forward_from_liger_state(x_2d, weight, bias, rstd):
    n_cols = x_2d.shape[-1]

    xf = x_2d.to(_torch.float32)
    wf = weight.to(_torch.float32)
    bf = bias.reshape(-1).to(_torch.float32)

    r3 = rstd[:, 0:1].to(_torch.float32)
    r2 = rstd[:, 1:2].to(_torch.float32)
    r1 = rstd[:, 2:3].to(_torch.float32)

    x2 = xf * xf
    x3 = x2 * xf

    y = (
        wf[0].reshape(1, n_cols) * (x3 * r3)
        + wf[1].reshape(1, n_cols) * (x2 * r2)
        + wf[2].reshape(1, n_cols) * (xf * r1)
        + bf.reshape(1, n_cols)
    )
    return y.to(dtype=x_2d.dtype)


def _poly_norm_backward_reference_order(dout_2d, x_2d, weight, bias, eps):
    del bias

    n_rows, n_cols = x_2d.shape
    inv_n = 1.0 / float(n_cols)

    xf = x_2d.to(_torch.float32)
    gf = dout_2d.to(_torch.float32)

    x2 = xf.pow(2)
    x3 = xf.pow(3)

    r1 = _torch.rsqrt(x2.mean(dim=1, keepdim=True) + eps)
    r2 = _torch.rsqrt(xf.pow(4).mean(dim=1, keepdim=True) + eps)
    r3 = _torch.rsqrt(xf.pow(6).mean(dim=1, keepdim=True) + eps)

    scalar_affine = weight.dim() == 1 and weight.numel() == 3

    if scalar_affine:
        wf = weight.to(_torch.float32).reshape(3)

        gw0 = gf * wf[0]
        gw1 = gf * wf[1]
        gw2 = gf * wf[2]

        dweight0 = (gf * (x3 * r3)).sum()
        dweight1 = (gf * (x2 * r2)).sum()
        dweight2 = (gf * (xf * r1)).sum()
        dweight = _torch.stack((dweight0, dweight1, dweight2)).reshape_as(weight)

        dbias = gf.sum()

    else:
        wf = weight.to(_torch.float32)

        w0 = wf[0].reshape(1, n_cols)
        w1 = wf[1].reshape(1, n_cols)
        w2 = wf[2].reshape(1, n_cols)

        gw0 = gf * w0
        gw1 = gf * w1
        gw2 = gf * w2

        dweight = _torch.empty((3, n_cols), dtype=_torch.float32, device=weight.device)
        dweight[0] = (gf * (x3 * r3)).sum(dim=0)
        dweight[1] = (gf * (x2 * r2)).sum(dim=0)
        dweight[2] = (gf * (xf * r1)).sum(dim=0)

        dbias = gf.sum(dim=0)

    s3 = (gw0 * x3).sum(dim=1, keepdim=True)
    s2 = (gw1 * x2).sum(dim=1, keepdim=True)
    s1 = (gw2 * xf).sum(dim=1, keepdim=True)

    x5 = xf.pow(5)

    r3_3 = r3 * r3 * r3
    r2_3 = r2 * r2 * r2
    r1_3 = r1 * r1 * r1

    dx3 = 3.0 * gw0 * x2 * r3 - (3.0 * inv_n) * r3_3 * x5 * s3
    dx2 = 2.0 * gw1 * xf * r2 - (2.0 * inv_n) * r2_3 * x3 * s2
    dx1 = gw2 * r1 - inv_n * r1_3 * xf * s1

    dx = dx3 + dx2 + dx1

    return dx, dweight, dbias


def make_liger_poly_norm_autograd_pair_fns():
    poly_norm_forward, poly_norm_backward = _load_liger()
    del poly_norm_backward

    def forward_with_saved(x, weight, bias, eps):
        with _torch.no_grad():
            x = x.contiguous()
            weight = weight.contiguous()
            bias = bias.contiguous()

            n_cols = x.shape[-1]

            if weight.dim() == 1 and weight.numel() == 3 and bias.numel() == 1:
                y, x_2d, rstd, _block_size, _num_warps = poly_norm_forward(
                    x,
                    weight,
                    bias.reshape(()),
                    eps,
                )
                return y.to(dtype=x.dtype), (x_2d, weight, bias, rstd)

            if weight.dim() != 2 or weight.shape[0] != 3 or weight.shape[1] != n_cols:
                raise ValueError("Liger PolyNorm wrapper expects weight shape (3, hidden_dim).")
            if bias.numel() != n_cols:
                raise ValueError("Liger PolyNorm wrapper expects bias.numel() == hidden_dim.")

            dummy_w = _torch.zeros((3,), dtype=_torch.float32, device=x.device)
            dummy_b = _torch.zeros((), dtype=_torch.float32, device=x.device)

            _unused_y, x_2d, rstd, _block_size, _num_warps = poly_norm_forward(
                x,
                dummy_w,
                dummy_b,
                eps,
            )

            y_2d = _poly_norm_vector_forward_from_liger_state(x_2d, weight, bias, rstd)
            return y_2d.view_as(x), (x_2d, weight, bias, rstd)

    def backward_from_saved(dout, saved_tensors, eps):
        with _torch.no_grad():
            x_2d, weight, bias, _rstd = saved_tensors

            dout = dout.contiguous()
            n_cols = x_2d.shape[-1]
            dout_2d = dout.view(-1, n_cols)

            dx, dweight, dbias = _poly_norm_backward_reference_order(
                dout_2d,
                x_2d,
                weight,
                bias,
                eps,
            )

            return (
                dx.to(dtype=x_2d.dtype).view_as(dout),
                dweight.to(dtype=weight.dtype).reshape_as(weight),
                dbias.to(dtype=bias.dtype).reshape_as(bias),
            )

    return forward_with_saved, backward_from_saved
