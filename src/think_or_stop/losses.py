"""Training objectives.

Multi-exit objective (deep supervision)::

    L_total = w1 * CE(exit1, y) + w2 * CE(exit2, y) + w3 * CE(exit3, y)

The default weights are (1, 1, 1): an un-normalised sum where every exit gets
an equal share of the gradient. Shared early stages therefore receive gradient
from all deeper heads plus their own. Weights are set with ``--exit-weights``.
For the baseline there is a single exit and the loss is plain cross-entropy.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class MultiExitLoss(nn.Module):
    """Weighted sum of per-exit cross-entropy losses."""

    def __init__(self, weights: Sequence[float]):
        super().__init__()
        if len(weights) == 0:
            raise ValueError("need at least one exit weight")
        self.weights = tuple(float(w) for w in weights)

    def forward(self, logits: Sequence[torch.Tensor], target: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if len(logits) != len(self.weights):
            raise ValueError(f"got {len(logits)} exits but {len(self.weights)} weights")
        per_exit = [F.cross_entropy(z.float(), target) for z in logits]
        total = sum(w * l for w, l in zip(self.weights, per_exit))
        return total, [l.detach() for l in per_exit]


def build_loss(model_type: str, exit_weights: Sequence[float]) -> MultiExitLoss:
    """Baseline: CE on the single final exit. Multi-exit: weighted sum."""
    return MultiExitLoss([1.0] if model_type == "baseline" else exit_weights)
