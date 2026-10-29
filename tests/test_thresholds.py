import math

import numpy as np
import pytest

from think_or_stop.inference import simulate_adaptive
from think_or_stop.metrics import (
    adaptive_summary,
    classification_report,
    confusion_matrix,
    expected_calibration_error,
)
from think_or_stop.models import ExitCosts
from think_or_stop.thresholds import select_threshold

COSTS = ExitCosts(stage_macs=[10, 20, 30], head_macs=[1, 1, 1])
STAGES = (0, 1, 2)


def _one_hot_probs(pred, conf, k=4):
    """Probabilities with argmax ``pred`` and max value ``conf``; rest spread evenly."""
    p = np.full((len(pred), k), 0.0)
    for i, (c, v) in enumerate(zip(pred, conf)):
        p[i] = (1 - v) / (k - 1)
        p[i, c] = v
    return p


def _val_set():
    """Exit 1 is right when confident (>=0.9) and wrong otherwise; exit 3 always right."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 4, n)
    conf1 = np.where(np.arange(n) < 100, 0.95, 0.6)
    pred1 = np.where(conf1 >= 0.9, y, (y + 1) % 4)
    e1 = _one_hot_probs(pred1, conf1)
    e2 = _one_hot_probs((y + 1) % 4, np.full(n, 0.5))
    e3 = _one_hot_probs(y, np.full(n, 0.99))
    return np.stack([e1, e2, e3], 1), y


def test_selection_relative_picks_cheapest_feasible():
    probs, y = _val_set()
    sel = select_threshold(probs, y, COSTS, STAGES, "relative", max_drop=0.0)
    assert sel.constraint_satisfied
    assert sel.reference_accuracy == 1.0
    # Thresholds in (0.6, 0.95] exit exactly the 100 confident samples with no loss.
    assert 0.6 < sel.selected_threshold <= 0.95
    assert sel.validation_result["accuracy"] == 1.0
    assert sel.validation_result["exit1_fraction"] == 0.5
    assert sel.data_split == "validation"


def test_selection_absolute_infeasible_falls_back_to_full_depth():
    probs, y = _val_set()
    probs[:, 2] = _one_hot_probs((y + 1) % 4, np.full(len(y), 0.99))  # final exit always wrong
    sel = select_threshold(probs, y, COSTS, STAGES, "absolute", min_accuracy=0.99)
    assert not sel.constraint_satisfied
    assert math.isinf(sel.selected_threshold)
    assert sel.validation_result["exit3_fraction"] == 1.0


def test_selection_depends_only_on_validation_inputs():
    probs, y = _val_set()
    a = select_threshold(probs, y, COSTS, STAGES, "relative", max_drop=0.05)
    b = select_threshold(probs.copy(), y.copy(), COSTS, STAGES, "relative", max_drop=0.05)
    assert a.selected_threshold == b.selected_threshold
    # The API has no test-set argument at all.
    import inspect

    params = inspect.signature(select_threshold).parameters
    assert not any("test" in p for p in params)


def test_selection_rejects_bad_arguments():
    probs, y = _val_set()
    with pytest.raises(ValueError):
        select_threshold(probs, y, COSTS, STAGES, "absolute")
    with pytest.raises(ValueError):
        select_threshold(probs, y, COSTS, STAGES, "bogus")
    with pytest.raises(ValueError):
        select_threshold(probs[:, -1:], y, COSTS, (2,), "relative")


def test_simulate_and_summary_costs():
    probs, y = _val_set()
    sim = simulate_adaptive(probs, 0.9)
    assert (sim.exit_indices[:100] == 0).all() and (sim.exit_indices[100:] == 2).all()
    s = adaptive_summary(probs, y, 0.9, COSTS, STAGES)
    # exit-1 cost = 10+1, exit-3 adaptive cost = 60+3
    assert s["mean_macs"] == pytest.approx(0.5 * 11 + 0.5 * 63)
    assert s["relative_macs"] == pytest.approx(s["mean_macs"] / 61)
    assert s["early_exit_error_rate"] == 0.0


def test_ece_and_confusion():
    conf = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([1, 1, 1, 0], dtype=bool)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.15)
    pred, tgt = np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2])
    cm = confusion_matrix(pred, tgt, 3)
    assert cm.tolist() == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    rep = classification_report(pred, tgt, ["a", "b", "c"])
    assert rep["c"]["recall"] == 0.5 and rep["b"]["precision"] == 0.5
