"""PyTorch reference for element-wise GeGLU (GELU-gated MLP activation).

Forward:
    c = gelu_tanh(a) * b

where gelu_tanh is the tanh approximation of GELU (the variant Liger's geglu kernel
implements):
    gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Inputs a and b are both [rows, cols].  This is the "gate * up" form used in GELU-gated
FFN layers.  The tanh approximation (not the exact erf GELU) is used so the reference
matches the Liger strong baseline exactly.
"""

import torch


def geglu_forward_ref(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """GeGLU forward: c = gelu_tanh(a) * b, preserving input dtype."""
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")
    return torch.nn.functional.gelu(a.float(), approximate="tanh").to(a.dtype) * b
