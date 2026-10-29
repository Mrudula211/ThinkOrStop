"""Generate PNG figures from the metrics written by ``evaluate`` and ``train``.

    python -m think_or_stop.plots --metrics-dir results/metrics/cifar10_multiexit_s0 \
        --baseline-dir results/metrics/cifar10_baseline_s0

Figures go to ``results/figures/<run>/``. Every figure is drawn from CSV/JSON
files that stay alongside it, so the plotted numbers can be inspected.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .utils import read_json  # noqa: E402

# Validated categorical palette (fixed order, light surface) and text/grid ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
EXIT_COLORS = SERIES[:3]  # exit 1, 2, 3 always keep the same colour
ADAPTIVE, SELECTED = SERIES[6], SERIES[7]  # threshold sweep / manual policy, val-selected policy

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 9.5, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0, "font.size": 9.5,
    "text.color": INK,
})


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _label(ax, x, y, text, dx=6, dy=4, color=INK2):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=color)


def _selected_threshold(metrics_dir: Path) -> float | None:
    p = metrics_dir / "threshold_selection.json"
    if not p.is_file():
        return None
    t = float(read_json(p)["selected_threshold"])
    return t if math.isfinite(t) else None


# --------------------------------------------------------------------------
# Individual figures
# --------------------------------------------------------------------------


def plot_accuracy_vs_threshold(mdir: Path, out: Path) -> Path:
    val, test = read_csv(mdir / "sweep_val.csv"), read_csv(mdir / "sweep_test.csv")
    sel = _selected_threshold(mdir)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for rows, name, c in ((val, "validation", SERIES[0]), (test, "test", SERIES[1])):
        t = [r["threshold"] for r in rows]
        ax.plot(t, [r["accuracy"] for r in rows], color=c, label=f"adaptive ({name})")
    ax.axhline(test[0]["full_depth_accuracy"], color=INK2, lw=1, ls="--")
    _label(ax, 0.0, test[0]["full_depth_accuracy"], "full depth (test)", dx=2, dy=-11)
    bx.plot([r["threshold"] for r in test], [r["relative_macs"] for r in test], color=SERIES[1], label="test")
    bx.plot([r["threshold"] for r in val], [r["relative_macs"] for r in val], color=SERIES[0], label="validation")
    for a in (ax, bx):
        if sel is not None:
            a.axvline(sel, color=INK, lw=1, ls=":")
    if sel is not None:
        _label(ax, sel, ax.get_ylim()[0], f"selected on val: t={sel:g}", dx=-4, dy=6)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs. confidence threshold")
    ax.legend(loc="center left")
    bx.set_ylabel("MACs / full-depth MACs")
    bx.set_xlabel("confidence threshold (max softmax probability)")
    bx.set_title("Estimated computation vs. threshold")
    return _save(fig, out / "accuracy_vs_threshold.png")


def plot_exit_distribution(mdir: Path, out: Path) -> Path:
    rows = read_csv(mdir / "sweep_test.csv")
    t = np.array([r["threshold"] for r in rows])
    fr = np.array([[r[f"exit{e}_fraction"] for e in (1, 2, 3)] for r in rows]).T
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.stackplot(t, fr, colors=EXIT_COLORS, labels=["Exit 1", "Exit 2", "Exit 3 (final)"],
                 edgecolor=SURFACE, linewidth=1.0)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("fraction of test images")
    ax.set_title("Where test images exit, by threshold")
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.18))
    sel = _selected_threshold(mdir)
    if sel is not None:
        ax.axvline(sel, color=INK, lw=1, ls=":")
        _label(ax, sel, 0.95, f"selected t={sel:g}", dx=-4, dy=0, color=INK)
    return _save(fig, out / "exit_distribution_vs_threshold.png")


def plot_compute_vs_accuracy(mdir: Path, out: Path, baseline_dir: Path | None) -> Path:
    sweep = read_csv(mdir / "sweep_test.csv")
    methods = read_csv(mdir / "methods_test.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot([r["relative_macs"] for r in sweep], [r["accuracy"] for r in sweep], color=ADAPTIVE,
            label="adaptive, threshold sweep (test)")
    fixed = [m for m in methods if m["method"].endswith("_only")]
    for i, m in enumerate(fixed):
        ax.plot(m["relative_macs"], m["accuracy"], "o", ms=8, color=EXIT_COLORS[i], mec=SURFACE, mew=2)
        _label(ax, m["relative_macs"], m["accuracy"], f"Exit {i + 1} only")
    for m in methods:
        if m["method"] == "adaptive_selected":
            ax.plot(m["relative_macs"], m["accuracy"], "D", ms=8, color=SELECTED, mec=SURFACE, mew=2,
                    label=f"val-selected policy (t={m['threshold']:g})")
    if baseline_dir is not None:
        b = [m for m in read_csv(baseline_dir / "methods_test.csv") if m["method"] == "standard_cnn"][0]
        ax.plot(b["relative_macs"], b["accuracy"], "s", ms=8, color=INK2, mec=SURFACE, mew=2, label="standard CNN")
    ax.set_xlabel("mean MACs per image / full-depth MACs (estimated)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs. estimated computation")
    ax.legend(loc="lower right")
    return _save(fig, out / "compute_vs_accuracy.png")


def plot_latency_vs_accuracy(mdir: Path, out: Path, baseline_dir: Path | None) -> Path | None:
    if not (mdir / "latency.csv").is_file():
        return None
    lat = read_csv(mdir / "latency.csv")
    sweep = {round(r["threshold"], 4): r["accuracy"] for r in read_csv(mdir / "sweep_test.csv")}
    acc = {m["method"]: m["accuracy"] for m in read_csv(mdir / "methods_test.csv")}
    base_lat = read_csv(baseline_dir / "latency.csv") if baseline_dir and (baseline_dir / "latency.csv").is_file() else []
    base_acc = {m["method"]: m["accuracy"] for m in read_csv(baseline_dir / "methods_test.csv")} if baseline_dir else {}

    def accuracy_of(r):
        if r["mode"] in acc:
            return acc[r["mode"]]
        t = r.get("threshold")
        return sweep.get(round(t, 4)) if isinstance(t, float) else None

    sizes = sorted({int(r["batch_size"]) for r in lat})
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.0 * len(sizes), 4.0), squeeze=False)
    for ax, bs in zip(axes[0], sizes):
        rows = [r for r in lat if int(r["batch_size"]) == bs]
        ad = sorted((r for r in rows if r["mode"].startswith("adaptive_t")), key=lambda r: r["threshold"])
        pts = [(r["per_sample_latency_ms_median"], accuracy_of(r), r["threshold"]) for r in ad]
        pts = [p for p in pts if p[1] is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=5, color=ADAPTIVE, label="adaptive (thresholds)")
            _label(ax, pts[0][0], pts[0][1], f"t={pts[0][2]:g}")
            _label(ax, pts[-1][0], pts[-1][1], f"t={pts[-1][2]:g}", dx=4, dy=-12)
        for i, r in enumerate(r for r in rows if r["mode"].endswith("_only")):
            ax.plot(r["per_sample_latency_ms_median"], acc[r["mode"]], "o", ms=8, color=EXIT_COLORS[i], mec=SURFACE, mew=2)
            _label(ax, r["per_sample_latency_ms_median"], acc[r["mode"]], f"Exit {i + 1} only", dx=-8, dy=8)
        for r in rows:
            if r["mode"] == "adaptive_selected":
                ax.plot(r["per_sample_latency_ms_median"], acc["adaptive_selected"], "D", ms=8, color=SELECTED,
                        mec=SURFACE, mew=2, label="val-selected policy")
        for r in base_lat:
            if int(r["batch_size"]) == bs and r["mode"] == "standard_cnn":
                ax.plot(r["per_sample_latency_ms_median"], base_acc["standard_cnn"], "s", ms=8, color=INK2,
                        mec=SURFACE, mew=2, label="standard CNN")
        ax.set_xlabel("median per-sample latency, ms" + (" (amortised over batch)" if bs > 1 else ""))
        ax.set_ylabel("test accuracy")
        ax.set_title(f"Latency vs. accuracy, batch size {bs} ({rows[0]['device']})")
        ax.legend(loc="lower right")
    return _save(fig, out / "latency_vs_accuracy.png")


def plot_calibration(mdir: Path, out: Path) -> Path:
    exits = read_json(mdir / "exits_test.json")
    n = len(exits)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False, sharey=True)
    offset = 3 - n
    for i, (ax, e) in enumerate(zip(axes[0], exits)):
        bins = [b for b in e["calibration_bins"] if b["count"] > 0]
        centers = [(b["lower"] + b["upper"]) / 2 for b in bins]
        width = bins[0]["upper"] - bins[0]["lower"] if bins else 0.066
        ax.bar(centers, [b["accuracy"] for b in bins], width=width * 0.9, color=EXIT_COLORS[i + offset],
               edgecolor=SURFACE, linewidth=1)
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Exit {i + 1 + offset}: ECE = {e['ece']:.3f}")
        ax.set_xlabel("confidence (15 equal-width bins)")
        if i == 0:
            ax.set_ylabel("accuracy in bin")
    fig.suptitle("Calibration (reliability diagrams, test set); dashed = perfect calibration", fontsize=10, color=INK2)
    fig.tight_layout()
    return _save(fig, out / "calibration_by_exit.png")


def plot_confusion(mdir: Path, out: Path, classes: Sequence[str] | None = None) -> Path:
    e = read_json(mdir / "exits_test.json")[-1]
    cm = np.array(e["confusion_matrix"], dtype=float)
    names = classes or list(e["per_class_accuracy"].keys())
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(f"Final-exit confusion matrix (test, row-normalised), acc = {e['accuracy']:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out / "confusion_matrix_final_exit.png")


def plot_corruption(mdir: Path, out: Path) -> Path | None:
    if not (mdir / "corruption.csv").is_file():
        return None
    rows = read_csv(mdir / "corruption.csv")
    conds = []
    for r in rows:
        key = (r["corruption"], int(r["severity"]))
        if key not in conds:
            conds.append(key)
    methods = [m for m in dict.fromkeys(r["method"] for r in rows)]
    colors = {m: (EXIT_COLORS[int(m[-1]) - 1] if m.startswith("exit") else None) for m in methods}
    extra = iter(SERIES[3:6])
    for m in methods:
        if colors[m] is None:
            colors[m] = SELECTED if m == "adaptive_selected" else ADAPTIVE if m.startswith("adaptive_t") else next(extra)
    labels = ["clean" if c == "none" else f"{c.replace('_', ' ')}\nsev {s}" for c, s in conds]
    source = rows[-1]["source"]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(max(7.0, 1.0 * len(conds) + 2), 6.4), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    w = 0.8 / len(methods)
    x = np.arange(len(conds))
    for k, m in enumerate(methods):
        acc = [next((r["accuracy"] for r in rows if r["method"] == m and (r["corruption"], int(r["severity"])) == c),
                    np.nan) for c in conds]
        ax.bar(x + (k - (len(methods) - 1) / 2) * w, acc, width=w * 0.92, color=colors[m], label=m.replace("_", " "))
    adaptive = [m for m in methods if m.startswith("adaptive")]
    for k, m in enumerate(adaptive):
        rel = [next((r["relative_macs"] for r in rows if r["method"] == m and (r["corruption"], int(r["severity"])) == c),
                    np.nan) for c in conds]
        bx.plot(x, rel, "-o", ms=6, color=colors[m], label=m.replace("_", " "))
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Clean vs. corrupted accuracy ({'synthetic corruptions' if source == 'synthetic' else 'CIFAR-10-C'}; "
                 "thresholds chosen on clean validation data)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    bx.set_ylabel("MACs / full depth")
    bx.set_ylim(0, 1.05)
    bx.set_xticks(x, labels, fontsize=7.5)
    bx.set_title("Estimated computation of adaptive policies" if adaptive else
                 "(no adaptive policies: single-exit model always uses 100% of MACs)")
    if adaptive:
        bx.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    return _save(fig, out / "clean_vs_corrupted.png")


def plot_training_curves(log_dir: Path, out: Path, name: str) -> Path | None:
    p = log_dir / "history.csv"
    if not p.is_file():
        return None
    h = read_csv(p)
    ep = [r["epoch"] for r in h]
    exits = sorted({k.split("exit")[-1] for k in h[0] if k.startswith("val_acc_exit")})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for e in exits:
        c = EXIT_COLORS[int(e) - 1]
        ax.plot(ep, [r[f"train_loss_exit{e}"] for r in h], color=c, ls="--", lw=1.5)
        ax.plot(ep, [r[f"val_loss_exit{e}"] for r in h], color=c, label=f"Exit {e}")
        bx.plot(ep, [r[f"train_acc_exit{e}"] for r in h], color=c, ls="--", lw=1.5)
        bx.plot(ep, [r[f"val_acc_exit{e}"] for r in h], color=c, label=f"Exit {e}")
    ax.set_title("Cross-entropy per exit (solid = val, dashed = train)")
    bx.set_title("Accuracy per exit (solid = val, dashed = train)")
    for a in (ax, bx):
        a.set_xlabel("epoch")
        a.legend()
    ax.set_ylabel("loss")
    bx.set_ylabel("accuracy")
    fig.suptitle(name, fontsize=10, color=INK2)
    fig.tight_layout()
    return _save(fig, out / "training_curves.png")


def generate_all(metrics_dir: Path, out_dir: Path, baseline_dir: Path | None = None) -> list[Path]:
    summary = read_json(metrics_dir / "summary.json")
    multi = summary["model"] == "multiexit"
    made: list[Path | None] = [
        plot_calibration(metrics_dir, out_dir),
        plot_confusion(metrics_dir, out_dir),
        plot_training_curves(Path(summary["log_dir"]), out_dir, summary["run_name"]),
        plot_corruption(metrics_dir, out_dir),
    ]
    if multi:
        made += [
            plot_accuracy_vs_threshold(metrics_dir, out_dir),
            plot_exit_distribution(metrics_dir, out_dir),
            plot_compute_vs_accuracy(metrics_dir, out_dir, baseline_dir),
            plot_latency_vs_accuracy(metrics_dir, out_dir, baseline_dir),
        ]
    return [p for p in made if p is not None]


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m think_or_stop.plots", description="Generate figures for one run.")
    p.add_argument("--metrics-dir", required=True, help="results/metrics/<run> written by evaluate")
    p.add_argument("--baseline-dir", default=None, help="metrics dir of a standard-CNN run to overlay")
    p.add_argument("--out-dir", default=None, help="default: results/figures/<run>")
    args = p.parse_args(argv)
    mdir = Path(args.metrics_dir)
    out = Path(args.out_dir) if args.out_dir else mdir.parent.parent / "figures" / mdir.name
    for path in generate_all(mdir, out, Path(args.baseline_dir) if args.baseline_dir else None):
        print("wrote", path)


if __name__ == "__main__":
    main()
