"""Classification, calibration and adaptive-inference metrics (NumPy only)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .inference import simulate_adaptive
from .models import ExitCosts

ECE_BINS = 15


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean()) if len(target) else float("nan")


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """``cm[i, j]`` = number of samples with true class i predicted as j."""
    return np.bincount(target * num_classes + pred, minlength=num_classes**2).reshape(num_classes, num_classes)


def classification_report(
    pred: np.ndarray, target: np.ndarray, class_names: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall (= per-class accuracy), F1 and support,
    plus macro and support-weighted averages."""
    k = len(class_names)
    cm = confusion_matrix(pred, target, k)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    report = {
        name: {"precision": precision[i], "recall": recall[i], "f1": f1[i], "support": int(support[i])}
        for i, name in enumerate(class_names)
    }
    w = support / support.sum()
    report["macro_avg"] = {"precision": precision.mean(), "recall": recall.mean(), "f1": f1.mean(), "support": int(support.sum())}
    report["weighted_avg"] = {
        "precision": float((precision * w).sum()),
        "recall": float((recall * w).sum()),
        "f1": float((f1 * w).sum()),
        "support": int(support.sum()),
    }
    return {n: {m: float(v) for m, v in d.items()} for n, d in report.items()}


def calibration_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS) -> list[dict[str, float]]:
    """Equal-width confidence bins over [0, 1]; bin i is (i/n, (i+1)/n].

    (The first bin also includes confidence 0.)"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = ids == b
        cnt = int(m.sum())
        bins.append(
            {
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": cnt,
                "mean_confidence": float(conf[m].mean()) if cnt else float("nan"),
                "accuracy": float(correct[m].mean()) if cnt else float("nan"),
            }
        )
    return bins


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)| with equal-width bins
    (Guo et al., 2017). Uses the max softmax probability as the confidence."""
    n = len(conf)
    if n == 0:
        return float("nan")
    ece = 0.0
    for b in calibration_bins(conf, correct, n_bins):
        if b["count"]:
            ece += b["count"] / n * abs(b["accuracy"] - b["mean_confidence"])
    return float(ece)


def exit_metrics(probs: np.ndarray, target: np.ndarray, class_names: Sequence[str]) -> list[dict]:
    """Fixed-exit metrics for each exit, from ``probs`` of shape (N, E, C)."""
    out = []
    for e in range(probs.shape[1]):
        p = probs[:, e]
        pred = p.argmax(1)
        conf = p.max(1)
        correct = pred == target
        report = classification_report(pred, target, class_names)
        out.append(
            {
                "exit": e + 1,
                "accuracy": accuracy(pred, target),
                "ece": expected_calibration_error(conf, correct),
                "mean_confidence": float(conf.mean()),
                "nll": float(-np.log(np.clip(p[np.arange(len(target)), target], 1e-12, None)).mean()),
                "per_class_accuracy": {c: report[c]["recall"] for c in class_names},
                "confusion_matrix": confusion_matrix(pred, target, len(class_names)),
                "classification_report": report,
                "calibration_bins": calibration_bins(conf, correct),
            }
        )
    return out


def adaptive_summary(
    probs: np.ndarray,
    target: np.ndarray,
    threshold: float | Sequence[float],
    costs: ExitCosts,
    exit_stages: tuple[int, ...],
) -> dict:
    """Accuracy, exit distribution, cost and reliability of the adaptive policy."""
    sim = simulate_adaptive(probs, threshold)
    n, num_exits, _ = probs.shape
    correct = sim.predictions == target
    per_exit_macs = np.array([costs.adaptive_exit_macs(e, exit_stages) for e in range(num_exits)], dtype=float)
    mean_macs = float(per_exit_macs[sim.exit_indices].mean())
    full = float(costs.full_macs)
    early = sim.exit_indices < num_exits - 1
    summary = {
        "threshold": threshold if isinstance(threshold, (int, float)) else list(threshold),
        "accuracy": accuracy(sim.predictions, target),
        "full_depth_accuracy": accuracy(probs[:, -1].argmax(1), target),
        "mean_macs": mean_macs,
        "mean_flops": 2 * mean_macs,
        "relative_macs": mean_macs / full,
        "mean_exit_depth": float((sim.exit_indices + 1).mean()),
        "early_exit_fraction": float(early.mean()),
        "early_exit_error_rate": float((~correct[early]).mean()) if early.any() else float("nan"),
        "ece": expected_calibration_error(sim.confidences, correct),
    }
    for e in range(num_exits):
        m = sim.exit_indices == e
        summary[f"exit{e + 1}_fraction"] = float(m.mean())
        summary[f"exit{e + 1}_accuracy"] = float(correct[m].mean()) if m.any() else float("nan")
    return summary


def threshold_sweep(
    probs: np.ndarray,
    target: np.ndarray,
    thresholds: Sequence[float],
    costs: ExitCosts,
    exit_stages: tuple[int, ...],
) -> list[dict]:
    return [adaptive_summary(probs, target, float(t), costs, exit_stages) for t in thresholds]
