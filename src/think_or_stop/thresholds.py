"""Validation-only confidence-threshold selection.

Algorithm
---------
1. Evaluate a grid of candidate thresholds ``t`` (one shared threshold for
   exits 1 and 2) on the **validation** outputs.
2. A candidate is feasible if it satisfies the constraint:

   * ``relative``: ``acc(t) >= acc_full - max_drop`` where ``acc_full`` is the
     validation accuracy of the *same* multi-exit model always using Exit 3.
   * ``absolute``: ``acc(t) >= min_accuracy``.

3. Among feasible candidates, choose the one with the lowest expected MACs per
   image (the computation proxy, see ``models.ExitCosts``). Ties are broken by
   higher validation accuracy, then by higher threshold (more conservative).
4. If nothing is feasible, fall back to ``threshold = inf`` (never exit early,
   i.e. full depth) and report ``constraint_satisfied = False``. In relative
   mode full depth always meets the constraint by definition, so the
   fallback can only occur in absolute mode.

The chosen threshold is then applied *unchanged* to the test set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .metrics import adaptive_summary, threshold_sweep
from .models import ExitCosts


def default_threshold_grid() -> list[float]:
    """0.00, 0.01, ..., 1.00 plus finer steps near 1, where most decisions happen."""
    grid = set(np.round(np.linspace(0.0, 1.0, 101), 4).tolist())
    grid.update([0.991, 0.993, 0.995, 0.997, 0.998, 0.999, 0.9995, 0.9999])
    return sorted(grid)


@dataclass
class ThresholdSelection:
    constraint: str  # "relative" or "absolute"
    target: float  # max_drop (relative) or min_accuracy (absolute)
    selected_threshold: float
    constraint_satisfied: bool
    reference_accuracy: float  # full-depth (exit-3) validation accuracy
    required_accuracy: float
    validation_result: dict
    sweep: list[dict] = field(repr=False, default_factory=list)
    data_split: str = "validation"

    def to_dict(self) -> dict:
        return asdict(self)


def select_threshold(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    costs: ExitCosts,
    exit_stages: tuple[int, ...],
    constraint: str = "relative",
    max_drop: float = 0.01,
    min_accuracy: float | None = None,
    grid: list[float] | None = None,
) -> ThresholdSelection:
    """Pick the cheapest threshold meeting an accuracy constraint on validation data.

    This function only ever receives validation arrays; callers must not pass
    test outputs.
    """
    if val_probs.shape[1] < 2:
        raise ValueError("threshold selection requires a model with intermediate exits")
    grid = grid if grid is not None else default_threshold_grid()
    reference = float((val_probs[:, -1].argmax(1) == val_targets).mean())
    if constraint == "relative":
        if max_drop < 0:
            raise ValueError("max_drop must be >= 0")
        required, target = reference - max_drop, max_drop
    elif constraint == "absolute":
        if min_accuracy is None or not 0.0 <= min_accuracy <= 1.0:
            raise ValueError("absolute constraint needs min_accuracy in [0, 1]")
        required, target = min_accuracy, min_accuracy
    else:
        raise ValueError(f"unknown constraint {constraint!r}; use 'relative' or 'absolute'")

    sweep = threshold_sweep(val_probs, val_targets, grid, costs, exit_stages)
    # Small epsilon so that e.g. a 0.01 drop computed in float is not rejected by rounding.
    feasible = [r for r in sweep if r["accuracy"] >= required - 1e-9]
    if feasible:
        best = min(feasible, key=lambda r: (r["mean_macs"], -r["accuracy"], -r["threshold"]))
        chosen, ok, result = best["threshold"], True, best
    else:
        chosen, ok = float("inf"), False
        result = adaptive_summary(val_probs, val_targets, chosen, costs, exit_stages)
    return ThresholdSelection(
        constraint=constraint,
        target=target,
        selected_threshold=chosen,
        constraint_satisfied=ok,
        reference_accuracy=reference,
        required_accuracy=required,
        validation_result=result,
        sweep=sweep,
    )
