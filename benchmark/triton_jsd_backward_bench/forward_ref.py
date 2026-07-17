"""PyTorch reference forward for jsd (used for AtenIR extraction and as the oracle)."""

import torch

def jsd_forward(log_q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    beta = 0.5
    rows = log_q.shape[0]
    log_p = target.float()
    log_qf = log_q.float()
    p = log_p.exp()
    q = log_qf.exp()
    m = beta * p + (1.0 - beta) * q
    log_m = m.log()
    kl_p_m = (p * (log_p - log_m)).sum()
    kl_q_m = (q * (log_qf - log_m)).sum()
    return (beta * kl_p_m + (1.0 - beta) * kl_q_m) / rows


jsd_forward_ref = jsd_forward

