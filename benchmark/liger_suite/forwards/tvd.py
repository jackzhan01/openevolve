"""PyTorch reference for Total Variation Distance loss.

Forward:
    loss = sum_ij 0.5 * |p_ij - q_ij| / BT        (reduction="batchmean")

matching Liger's `LigerTVDLossFunction` with the pinned config: reduction="batchmean",
no shift_labels / ignore_index. `p` is the student distribution (differentiable),
`q` is the teacher TARGET distribution (probabilities, NO gradient) — same convention
as the kl_div benchmark case.

Inputs:
    p : [rows, cols]  probabilities, each row sums to 1 (differentiable)
    q : [rows, cols]  probabilities, each row sums to 1 (target, no gradient)
"""

import torch


def tvd_forward(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    rows = p.shape[0]
    return (0.5 * (p.float() - q.float()).abs()).sum() / rows
