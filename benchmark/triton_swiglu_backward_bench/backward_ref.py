"""Pure-PyTorch backward reference for SwiGLU.

Given upstream gradient dc and forward inputs a, b:
    sig_a  = sigmoid(a)
    silu_a = a * sig_a
    db     = dc * silu_a
    da     = dc * b * (silu_a * (1 - sig_a) + sig_a)

All computation is done in float32 with output cast to match input dtypes.
"""

import torch


def swiglu_backward_ref(
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (da, db) for c = silu(a) * b."""
    af = a.float()
    bf = b.float()
    dcf = dc.float()
    sig_a = torch.sigmoid(af)
    silu_a = af * sig_a
    da = (dcf * bf * (silu_a * (1.0 - sig_a) + sig_a)).to(a.dtype)
    db = (dcf * silu_a).to(b.dtype)
    return da, db
