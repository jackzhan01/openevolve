"""PyTorch reference forward for tvd (used for AtenIR extraction and as the oracle)."""

import torch

def tvd_forward(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    rows = p.shape[0]
    return (0.5 * (p.float() - q.float()).abs()).sum() / rows


tvd_forward_ref = tvd_forward

