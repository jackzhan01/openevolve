"""Triton layout kernels embedded into Pipeline B autograd-pair seeds.

AtenIR backward graphs of attention-style ops are dominated by layout nodes
(permute / expand / clone). atenir's dispatch lowers those to PyTorch-native
calls, which is fine for in-process verification but leaves the emitted seed
mostly opaque to OpenEvolve — it can only rewrite Triton source inside the
EVOLVE-BLOCK. ``program_codegen`` therefore substitutes the launchers below for
those nodes at seed-emission time and embeds this module's source (everything
after the EMBED-START marker) into the seed's kernel library.

atenir itself is untouched: its dispatch registry keeps the PyTorch fallbacks.

One rank-8-padded gather-copy kernel covers all three ops: each output linear
index is decomposed into coordinates against the (contiguous) output shape, and
the source offset is the stride-weighted sum of those coordinates. Broadcast
(expand) dims carry stride 0.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - kernels are only launched on CUDA
    triton = None
    tl = None

# EMBED-START


@triton.jit
def _strided_copy_kernel(
    src_ptr,
    out_ptr,
    N,
    s0, s1, s2, s3, s4, s5, s6, s7,  # output sizes, rank padded to 8
    t0, t1, t2, t3, t4, t5, t6, t7,  # source strides (elements), 0 = broadcast
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    rem = offs
    src_off = tl.zeros_like(offs)
    idx = rem % s7
    rem = rem // s7
    src_off += idx * t7
    idx = rem % s6
    rem = rem // s6
    src_off += idx * t6
    idx = rem % s5
    rem = rem // s5
    src_off += idx * t5
    idx = rem % s4
    rem = rem // s4
    src_off += idx * t4
    idx = rem % s3
    rem = rem // s3
    src_off += idx * t3
    idx = rem % s2
    rem = rem // s2
    src_off += idx * t2
    idx = rem % s1
    rem = rem // s1
    src_off += idx * t1
    idx = rem % s0
    src_off += idx * t0
    val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


def strided_copy(a, out_shape, src_strides):
    """Materialize a contiguous tensor of ``out_shape`` gathered from ``a``.

    ``src_strides[i]`` is the element stride of ``a`` along output dim ``i``
    (0 for broadcast dims). Rank up to 8.
    """
    out_shape = [int(s) for s in out_shape]
    src_strides = [int(s) for s in src_strides]
    out = torch.empty(out_shape, device=a.device, dtype=a.dtype)
    n = out.numel()
    if n == 0:
        return out
    pad = 8 - len(out_shape)
    sizes = [1] * pad + out_shape
    strides = [0] * pad + src_strides
    grid = ((n + 1023) // 1024,)
    _strided_copy_kernel[grid](a, out, n, *sizes, *strides, BLOCK=1024)
    return out


def permute_copy(a, dims):
    """Contiguous permute: equivalent to a.permute(dims).contiguous()."""
    dims = [int(d) % a.ndim for d in dims]
    return strided_copy(a, [a.shape[d] for d in dims], [a.stride(d) for d in dims])


def expand_copy(a, shape):
    """Contiguous expand: equivalent to a.expand(shape).contiguous().

    Follows aten.expand semantics: ``shape`` may prepend new leading dims, and
    -1 keeps the source size. Size-1 source dims broadcast with stride 0.
    """
    shape = [int(s) for s in shape]
    lead = len(shape) - a.ndim
    out_shape, strides = [], []
    for i, s in enumerate(shape):
        if i < lead:
            out_shape.append(s)
            strides.append(0)
            continue
        src_dim = i - lead
        src_size = a.shape[src_dim]
        size = src_size if s == -1 else s
        out_shape.append(size)
        strides.append(0 if (src_size == 1 and size != 1) else a.stride(src_dim))
    return strided_copy(a, out_shape, strides)


def clone_copy(a):
    """Contiguous clone via the strided-copy kernel."""
    return strided_copy(a, list(a.shape), list(a.stride()))


# EMBED-END


def embeddable_source() -> str:
    """Source text between the EMBED marker lines, for inlining into a seed."""
    from pathlib import Path

    lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    start = lines.index("# EMBED-START") + 1
    end = lines.index("# EMBED-END")
    return "\n".join(lines[start:end]).strip() + "\n"
