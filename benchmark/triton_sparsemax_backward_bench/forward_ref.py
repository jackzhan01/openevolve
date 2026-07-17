"""PyTorch reference forward for sparsemax (used for AtenIR extraction and as the oracle)."""

import torch

def sparsemax_forward(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    z, _ = torch.sort(xf, dim=-1, descending=True)
    cumsum = z.cumsum(dim=-1)
    k = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=xf.dtype)
    support = (1.0 + k * z) > cumsum
    k_supp = support.to(xf.dtype).sum(dim=-1, keepdim=True)
    cum_supp = torch.where(support, z, torch.zeros_like(z)).sum(dim=-1, keepdim=True)
    tau = (cum_supp - 1.0) / k_supp
    return torch.clamp(xf - tau, min=0.0).to(x.dtype)


sparsemax_forward_ref = sparsemax_forward

