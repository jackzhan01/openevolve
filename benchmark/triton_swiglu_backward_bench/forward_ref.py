"""PyTorch reference for element-wise SwiGLU (SiLU-gated MLP activation).

Forward:
    c = silu(a) * b    where silu(x) = x * sigmoid(x)

Inputs a and b are both [rows, cols].  This is the "gate * up" form used
in LLaMA-style FFN layers.  The gate_multiplier is fixed at 1.0.
"""

import torch


def swiglu_forward_ref(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU forward: c = silu(a) * b, preserving input dtype."""
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")
    return torch.nn.functional.silu(a.float()).to(a.dtype) * b
