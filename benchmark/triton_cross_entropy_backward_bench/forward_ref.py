"""PyTorch reference for Cross-Entropy loss (hard label, mean reduction).

Forward:
    loss = mean_reduction( CrossEntropy(logits, target) )
         = mean_i( -log_softmax(logits_i)[target_i] )

Inputs:
    logits : [BT, V]  (float)   -- BT = batch*seq flattened, V = vocab size
    target : [BT]     (int64)   -- class indices in [0, V)

This matches Liger's cross_entropy kernel configuration we benchmark against:
hard labels, no label smoothing, no z-loss, no class weighting, ignore_index=-100,
reduction="mean". The loss is computed in fp32 internally and returned in the logits dtype.
"""

import torch


def cross_entropy_forward_ref(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean-reduced cross-entropy loss (scalar), preserving logits dtype."""
    loss = torch.nn.functional.cross_entropy(
        logits.float(), target, reduction="mean", ignore_index=-100
    )
    return loss.to(logits.dtype)
