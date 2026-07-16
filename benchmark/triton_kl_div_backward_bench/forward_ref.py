"""PyTorch reference for KL-divergence loss (batchmean reduction, non-log target).

Forward:
    loss = KL(target || input) = sum_ij target_ij * (log target_ij - input_ij) / BT

where `input` (y_pred) is log-probabilities and `target` (y_true) is probabilities, matching
Liger's kl_div kernel config (reduction="batchmean", log_target=False). Backward is only wrt
the input log-probs: d_input = -target / BT (scaled by the loss cotangent).

Inputs:
    y_pred : [BT, V]  log-probabilities (e.g. log_softmax of logits)
    y_true : [BT, V]  probabilities     (e.g. softmax of logits), each row sums to 1
"""

import torch


def kl_div_forward_ref(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Batchmean KL divergence loss (scalar). y_pred are log-probs, y_true are probs."""
    return torch.nn.functional.kl_div(y_pred, y_true, reduction="batchmean", log_target=False)
