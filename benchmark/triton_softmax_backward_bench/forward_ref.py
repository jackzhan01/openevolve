"""PyTorch reference for row-wise Softmax (over the last dim).

Forward:
    y = softmax(x, dim=-1)

Inputs x are [rows, cols]; softmax is taken over the last dim (cols). Matches Liger's
softmax kernel (numerically-stable softmax over the last dimension). Computed in fp32
internally and returned in the input dtype.
"""

import torch


def softmax_forward_ref(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dim, preserving input dtype."""
    return torch.softmax(x.float(), dim=-1).to(x.dtype)
