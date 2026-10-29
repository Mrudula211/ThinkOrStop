"""Fixed-exit and confidence-based adaptive inference.

Adaptive policy (per sample):
    1. At each intermediate exit k, compute ``softmax(logits_k)``.
    2. Confidence = maximum class probability.
    3. If confidence >= threshold_k, return that exit's argmax and stop.
    4. Otherwise run the next stage. Exit 3 always returns a prediction.

The max softmax probability is only a *proxy* for correctness; a confident
early exit can still be wrong. In batched mode, samples that exit are removed
from the batch, so deeper stages run only on the still-active subset.

``simulate_adaptive`` evaluates the same policy from pre-computed logits of
all exits. Because a sample's exit-k logits do not depend on which other
samples are in the batch (eval-mode BN uses running statistics), it yields the
same decisions as the dynamic implementation up to floating-point noise from
different kernel choices; ``evaluate.py`` reports the measured agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .models import ThinkOrStopCNN


def _thresholds(threshold: float | Sequence[float], num_exits: int) -> list[float]:
    """Expand a scalar threshold to one value per intermediate exit."""
    n_inter = num_exits - 1
    if isinstance(threshold, (int, float)):
        return [float(threshold)] * n_inter
    t = [float(v) for v in threshold]
    if len(t) != n_inter:
        raise ValueError(f"need {n_inter} thresholds (one per intermediate exit), got {len(t)}")
    return t


@dataclass
class AdaptiveResult:
    predictions: torch.Tensor  # (N,) int64 class index
    exit_indices: torch.Tensor  # (N,) int64, 0-based exit position
    confidences: torch.Tensor  # (N,) max softmax prob at the chosen exit
    stage_batch_sizes: list[int]  # samples processed by each stage


@torch.inference_mode()
def adaptive_predict(
    model: ThinkOrStopCNN, x: torch.Tensor, threshold: float | Sequence[float] = 0.9
) -> AdaptiveResult:
    """Confidence-based early-exit inference on a batch (or a single sample).

    The model must be in eval mode (asserted) so that predictions are
    deterministic. Accepts ``(N, C, H, W)`` or a single ``(C, H, W)`` image.
    """
    if model.training:
        raise RuntimeError("adaptive_predict requires model.eval()")
    if x.dim() == 3:
        x = x.unsqueeze(0)
    thr = _thresholds(threshold, model.num_exits)
    n = x.size(0)
    device = x.device
    preds = torch.empty(n, dtype=torch.long, device=device)
    exits = torch.empty(n, dtype=torch.long, device=device)
    confs = torch.empty(n, dtype=torch.float32, device=device)
    active = torch.arange(n, device=device)  # original positions of active samples
    stage_sizes = [0] * len(model.stages)
    exit_of_stage = {s: i for i, s in enumerate(model.exit_stages)}
    last_exit = model.num_exits - 1

    h = x
    for s, stage in enumerate(model.stages):
        stage_sizes[s] = h.size(0)
        h = stage(h)
        if s not in exit_of_stage:
            continue
        e = exit_of_stage[s]
        probs = F.softmax(model.heads[e](h).float(), dim=1)
        conf, pred = probs.max(dim=1)
        if e == last_exit:
            done, n_done = None, h.size(0)
        else:
            done = conf >= thr[e]
            # One host sync per exit; skip mask indexing when no sample or every sample exits here.
            n_done = int(done.sum())
        if n_done == h.size(0):  # all remaining samples stop here; skip deeper stages
            preds[active] = pred
            exits[active] = e
            confs[active] = conf
            break
        if n_done == 0:
            continue
        idx = active[done]
        preds[idx] = pred[done]
        exits[idx] = e
        confs[idx] = conf[done]
        keep = ~done
        active = active[keep]
        h = h[keep]
    return AdaptiveResult(preds, exits, confs, stage_sizes)


@torch.inference_mode()
def fixed_exit_predict(model: ThinkOrStopCNN, x: torch.Tensor, exit_idx: int) -> torch.Tensor:
    """Logits from a single exit, computing only the stages it needs."""
    if model.training:
        raise RuntimeError("fixed_exit_predict requires model.eval()")
    return model.forward_to_exit(x, exit_idx)


# --------------------------------------------------------------------------
# Policy simulation from cached per-exit outputs
# --------------------------------------------------------------------------


@dataclass
class SimulatedPolicy:
    predictions: np.ndarray  # (N,)
    exit_indices: np.ndarray  # (N,)
    confidences: np.ndarray  # (N,)


def simulate_adaptive(probs: np.ndarray, threshold: float | Sequence[float]) -> SimulatedPolicy:
    """Apply the early-exit rule to cached softmax outputs.

    Args:
        probs: array ``(N, E, C)`` of per-exit class probabilities.
        threshold: scalar or one value per intermediate exit.
    """
    n, num_exits, _ = probs.shape
    thr = _thresholds(threshold, num_exits)
    conf = probs.max(axis=2)  # (N, E)
    pred = probs.argmax(axis=2)  # (N, E)
    exit_idx = np.full(n, num_exits - 1, dtype=np.int64)
    undecided = np.ones(n, dtype=bool)
    for e in range(num_exits - 1):
        take = undecided & (conf[:, e] >= thr[e])
        exit_idx[take] = e
        undecided &= ~take
    rows = np.arange(n)
    return SimulatedPolicy(pred[rows, exit_idx], exit_idx, conf[rows, exit_idx])
