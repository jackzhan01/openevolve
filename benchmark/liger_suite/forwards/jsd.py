"""PyTorch reference for the generalized Jensen-Shannon Divergence loss (beta=0.5).

Forward (Liger's convention — both arguments in LOG-space):
    P = exp(target)      # teacher probs
    Q = exp(log_q)       # student probs
    M = beta*P + (1-beta)*Q
    loss = ( beta * KL(P || M) + (1-beta) * KL(Q || M) ) / rows

matching Liger's `LigerJSDFunction` with the pinned config: beta=0.5, no shift_labels /
ignore_index (normalization is then by the number of rows). `log_q` is the student's
log-probabilities (differentiable); `target` is the teacher's log-probabilities
(NO gradient).

Inputs:
    log_q  : [rows, cols]  log-probabilities of the student (differentiable)
    target : [rows, cols]  log-probabilities of the teacher (target, no gradient)
"""

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
