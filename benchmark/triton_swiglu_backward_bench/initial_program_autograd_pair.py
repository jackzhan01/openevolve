"""Autograd-pair SwiGLU seed with an evolvable saved-tensor contract.

Public contract:

    swiglu_forward_with_saved(a, b) -> (c, saved_tensors)
    swiglu_backward_from_saved(dc, saved_tensors) -> (da, db)

This placeholder uses pure PyTorch (no Triton).  It is intended to be
replaced by the output of the autograd-pair fusion agent (Pipeline A/B/C).
OpenEvolve should then optimise the EVOLVE-BLOCK into Triton kernels.

NOTE: This file is the canonical seed for OpenEvolve.  Do NOT edit it
manually after the pipeline generates its version.
"""

from __future__ import annotations

import torch


# EVOLVE-BLOCK-START
def swiglu_forward_with_saved(a: torch.Tensor, b: torch.Tensor):
    """Forward pass: c = silu(a) * b.  Saves (a, b) for backward."""
    af = a.float()
    bf = b.float()
    sig_a = torch.sigmoid(af)
    silu_a = af * sig_a
    c = (silu_a * bf).to(a.dtype)
    # Conservative baseline: save original inputs.
    # Evolution may choose to save silu_a or sig_a instead.
    return c, (a, b)


def swiglu_backward_from_saved(
    dc: torch.Tensor,
    saved_tensors: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward pass using saved (a, b)."""
    a, b = saved_tensors
    af = a.float()
    bf = b.float()
    dcf = dc.float()
    sig_a = torch.sigmoid(af)
    silu_a = af * sig_a
    da = (dcf * bf * (silu_a * (1.0 - sig_a) + sig_a)).to(a.dtype)
    db = (dcf * silu_a).to(b.dtype)
    return da, db
# EVOLVE-BLOCK-END
