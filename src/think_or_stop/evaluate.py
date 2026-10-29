"""Evaluate a trained checkpoint: per-exit metrics, adaptive inference,
validation-only threshold selection, corruption robustness and latency.

Example::

    python -m think_or_stop.evaluate --checkpoint checkpoints/cifar10_multiexit_s0/best.pt \
        --device cuda --threshold 0.9 --select-threshold --max-drop 0.01 --corruptions synthetic

Evaluation discipline:
* Thresholds are selected on the validation split only, then applied
  unchanged to the clean test set and to every corrupted test set.
* Test-set threshold sweeps are written for *descriptive* trade-off plots and
  are never used to choose anything.

Outputs go to ``results/metrics/<run>/`` (override with ``--output-dir``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .benchmark import benchmark_modes
from .config import DATASETS, TrainConfig
from .datasets import (
    CIFAR10C,
    CIFAR10C_CORRUPTIONS,
    SYNTHETIC_CORRUPTIONS,
    build_splits,
    build_synthetic_corruption,
    make_loader,
)
from .inference import adaptive_predict, fixed_exit_predict, simulate_adaptive
from .metrics import adaptive_summary, exit_metrics, threshold_sweep
from .models import ExitCosts, ThinkOrStopCNN, build_model, compute_exit_costs, count_parameters
from .thresholds import default_threshold_grid, select_threshold
from .utils import environment_report, get_logger, load_checkpoint, resolve_device, write_csv, write_json

DEFAULT_LATENCY_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m think_or_stop.evaluate",
        description="Evaluate a checkpoint (fixed exits, adaptive inference, thresholds, robustness, latency).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint produced by train")
    p.add_argument("--dataset", default=None, help="sanity check: must match the checkpoint's dataset")
    p.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu")
    p.add_argument("--data-dir", default=None, help="dataset root (default: the one used in training)")
    p.add_argument("--batch-size", type=int, default=256, help="batch size for accuracy evaluation")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--output-dir", default=None, help="metrics directory (default: results/metrics/<run>[_<tag>])")
    p.add_argument("--tag", default="", help="suffix for the default output directory")
    g = p.add_argument_group("adaptive inference (multi-exit models only)")
    g.add_argument("--threshold", type=float, default=0.9, help="manual confidence threshold for exits 1 and 2")
    g.add_argument("--select-threshold", action=argparse.BooleanOptionalAction, default=True,
                   help="also select a threshold on the validation split")
    g.add_argument("--constraint", choices=("relative", "absolute"), default="relative",
                   help="relative: val acc >= full-depth val acc - max-drop; absolute: val acc >= min-accuracy")
    g.add_argument("--max-drop", type=float, default=0.01, help="allowed accuracy drop for --constraint relative")
    g.add_argument("--min-accuracy", type=float, default=None, help="required accuracy for --constraint absolute")
    g = p.add_argument_group("robustness")
    g.add_argument("--corruptions", choices=("none", "synthetic", "cifar10c"), default="none",
                   help="synthetic = reproducible Gaussian noise/blur on the test set; cifar10c = real CIFAR-10-C")
    g.add_argument("--severities", type=int, nargs="+", default=[1, 3, 5], help="corruption severities (1..5)")
    g.add_argument("--cifar10c-corruptions", nargs="+", default=list(CIFAR10C_CORRUPTIONS),
                   help="subset of CIFAR-10-C corruptions to evaluate")
    g = p.add_argument_group("latency")
    g.add_argument("--latency", action=argparse.BooleanOptionalAction, default=True, help="measure latency")
    g.add_argument("--latency-batch-sizes", type=int, nargs="+", default=[1, 128])
    g.add_argument("--latency-samples", type=int, default=1024, help="test images used for timing")
    g.add_argument("--latency-warmup", type=int, default=20, help="untimed warm-up batches per mode")
    g.add_argument("--latency-repeats", type=int, default=5, help="passes over the timing images per mode")
    g.add_argument("--latency-thresholds", type=float, nargs="*", default=list(DEFAULT_LATENCY_THRESHOLDS),
                   help="extra thresholds timed in adaptive mode (the manual/selected ones are always timed)")
    return p


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[ThinkOrStopCNN, TrainConfig, dict]:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = TrainConfig.from_dict(ckpt["config"])
    spec = cfg.spec
    model = build_model(spec.in_channels, cfg.model, spec.num_classes, cfg.base_channels, cfg.dropout)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg, ckpt


@torch.inference_mode()
def collect_probs(model: ThinkOrStopCNN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Softmax outputs of every exit: ``probs`` (N, E, C) float32 and targets (N,)."""
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(torch.stack([F.softmax(z.float(), dim=1) for z in logits], dim=1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy().astype(np.int64)


@torch.inference_mode()
def run_dynamic(model: ThinkOrStopCNN, loader: DataLoader, device: torch.device, threshold: float) -> dict[str, np.ndarray]:
    """Run the real batched early-exit implementation over a loader."""
    preds, exits, confs, targets = [], [], [], []
    stage_counts = np.zeros(len(model.stages), dtype=np.int64)
    for x, y in loader:
        r = adaptive_predict(model, x.to(device, non_blocking=True), threshold)
        preds.append(r.predictions.cpu())
        exits.append(r.exit_indices.cpu())
        confs.append(r.confidences.cpu())
        targets.append(y)
        stage_counts += np.array(r.stage_batch_sizes)
    return {
        "predictions": torch.cat(preds).numpy(),
        "exit_indices": torch.cat(exits).numpy(),
        "confidences": torch.cat(confs).numpy(),
        "targets": torch.cat(targets).numpy(),
        "stage_counts": stage_counts,
    }


def dynamic_check(model, loader, device, probs, threshold) -> dict:
    """Compare the dynamic implementation against the cached-logit simulation."""
    dyn = run_dynamic(model, loader, device, threshold)
    sim = simulate_adaptive(probs, threshold)
    return {
        "threshold": threshold,
        "dynamic_accuracy": float((dyn["predictions"] == dyn["targets"]).mean()),
        "prediction_agreement": float((dyn["predictions"] == sim.predictions).mean()),
        "exit_agreement": float((dyn["exit_indices"] == sim.exit_indices).mean()),
        "samples_processed_per_stage": dyn["stage_counts"].tolist(),
    }


def exit_rows(metrics: list[dict], costs: ExitCosts, exit_stages: tuple[int, ...], split: str) -> list[dict]:
    offset = 3 - len(metrics)
    rows = []
    for i, m in enumerate(metrics):
        macs = costs.fixed_exit_macs(i, exit_stages)
        rows.append({
            "split": split, "exit": i + 1 + offset, "accuracy": m["accuracy"], "ece": m["ece"],
            "nll": m["nll"], "mean_confidence": m["mean_confidence"], "macs": macs,
            "relative_macs": macs / costs.full_macs,
        })
    return rows


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    model, cfg, ckpt = load_model(args.checkpoint, device)
    if args.dataset and args.dataset != cfg.dataset:
        raise ValueError(f"--dataset {args.dataset} does not match checkpoint dataset {cfg.dataset}")
    out_dir = Path(args.output_dir) if args.output_dir else (
        Path(cfg.output_dir) / "results" / "metrics" / (cfg.run_name + (f"_{args.tag}" if args.tag else ""))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("think_or_stop.evaluate", out_dir / "evaluate.log")
    env = environment_report(device)
    log.info("Environment: " + ", ".join(f"{k}={v}" for k, v in env.items()))
    log.info(f"Checkpoint {args.checkpoint}: {cfg.model} on {cfg.dataset}, epoch {ckpt['epoch']}, "
             f"best val {ckpt['select_metric']}={ckpt['best_score']:.4f}")
    log.info(f"Model parameters on {next(model.parameters()).device}; model.training={model.training}")

    spec = DATASETS[cfg.dataset]
    data_dir = args.data_dir or cfg.data_dir
    splits = build_splits(cfg.dataset, data_dir, cfg.split_seed, download=False)
    val_loader = make_loader(splits.val, args.batch_size, False, args.num_workers, device)
    test_loader = make_loader(splits.test, args.batch_size, False, args.num_workers, device)
    costs = compute_exit_costs(model, (spec.in_channels, spec.image_size, spec.image_size))
    es = model.exit_stages
    multi = model.num_exits > 1

    # ---- Fixed-exit metrics (Model A / Model C) -------------------------
    val_probs, val_y = collect_probs(model, val_loader, device)
    test_probs, test_y = collect_probs(model, test_loader, device)
    val_exits = exit_metrics(val_probs, val_y, spec.classes)
    test_exits = exit_metrics(test_probs, test_y, spec.classes)
    write_json(out_dir / "exits_val.json", val_exits)
    write_json(out_dir / "exits_test.json", test_exits)
    rows = exit_rows(val_exits, costs, es, "val") + exit_rows(test_exits, costs, es, "test")
    write_csv(out_dir / "exits.csv", rows)
    for r in rows:
        if r["split"] == "test":
            log.info(f"[test] exit {r['exit']}: acc={r['accuracy']:.4f} ece={r['ece']:.4f} "
                     f"MACs={r['macs']:,} ({r['relative_macs']:.1%} of full)")

    methods: list[dict] = []
    offset = 3 - model.num_exits
    for i, m in enumerate(test_exits):
        name = "standard_cnn" if not multi else f"multiexit_exit{i + 1 + offset}_only"
        methods.append({
            "method": name, "accuracy": m["accuracy"], "ece": m["ece"],
            "mean_macs": costs.fixed_exit_macs(i, es),
            "relative_macs": costs.fixed_exit_macs(i, es) / costs.full_macs,
            "mean_exit_depth": float(es[i] + 1),
        })

    summary: dict = {
        "run_name": cfg.run_name, "dataset": cfg.dataset, "model": cfg.model, "seed": cfg.seed,
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": ckpt["epoch"],
        "log_dir": str(cfg.log_dir), "parameters": count_parameters(model),
        "exit_costs": {"stage_macs": costs.stage_macs, "head_macs": costs.head_macs, "full_macs": costs.full_macs},
        "environment": env, "methods": methods,
    }

    # ---- Adaptive inference (Model D) -----------------------------------
    policies: dict[str, float] = {}
    if multi:
        grid = default_threshold_grid()
        write_csv(out_dir / "sweep_val.csv", threshold_sweep(val_probs, val_y, grid, costs, es))
        write_csv(out_dir / "sweep_test.csv", threshold_sweep(test_probs, test_y, grid, costs, es))
        policies[f"adaptive_t{args.threshold:g}"] = args.threshold

        if args.select_threshold:
            sel = select_threshold(val_probs, val_y, costs, es, args.constraint, args.max_drop, args.min_accuracy)
            write_json(out_dir / "threshold_selection.json", sel.to_dict())
            log.info(
                f"[val] threshold selection ({sel.constraint}, target={sel.target}): t={sel.selected_threshold} "
                f"satisfied={sel.constraint_satisfied} val acc={sel.validation_result['accuracy']:.4f} "
                f"(reference {sel.reference_accuracy:.4f}, required {sel.required_accuracy:.4f}), "
                f"relative MACs={sel.validation_result['relative_macs']:.3f}"
            )
            if not sel.constraint_satisfied:
                log.info("WARNING: no threshold met the constraint; falling back to full depth (no early exits)")
            summary["threshold_selection"] = {k: v for k, v in sel.to_dict().items() if k != "sweep"}
            policies["adaptive_selected"] = sel.selected_threshold

        adaptive = {}
        for name, t in policies.items():
            s = adaptive_summary(test_probs, test_y, t, costs, es)
            s["dynamic_check"] = dynamic_check(model, test_loader, device, test_probs, t)
            adaptive[name] = s
            methods.append({
                "method": name, "threshold": t, "accuracy": s["accuracy"], "ece": s["ece"],
                "mean_macs": s["mean_macs"], "relative_macs": s["relative_macs"],
                "mean_exit_depth": s["mean_exit_depth"],
                "exit1_fraction": s["exit1_fraction"], "exit2_fraction": s["exit2_fraction"],
                "exit3_fraction": s["exit3_fraction"], "early_exit_error_rate": s["early_exit_error_rate"],
            })
            dc = s["dynamic_check"]
            log.info(
                f"[test] {name} (t={t}): acc={s['accuracy']:.4f} (full depth {s['full_depth_accuracy']:.4f}) "
                f"exits={s['exit1_fraction']:.1%}/{s['exit2_fraction']:.1%}/{s['exit3_fraction']:.1%} "
                f"mean depth={s['mean_exit_depth']:.2f} relative MACs={s['relative_macs']:.3f} "
                f"early-exit error={s['early_exit_error_rate']:.4f} ECE={s['ece']:.4f} | dynamic impl: "
                f"acc={dc['dynamic_accuracy']:.4f} agreement={dc['prediction_agreement']:.4%} "
                f"stage counts={dc['samples_processed_per_stage']}"
            )
        write_json(out_dir / "adaptive_test.json", adaptive)

    # ---- Corruption robustness -------------------------------------------
    if args.corruptions != "none":
        summary["corruption"] = evaluate_corruptions(args, model, cfg, data_dir, device, costs, policies, test_probs,
                                                     test_y, out_dir, log)

    # ---- Latency ----------------------------------------------------------
    if args.latency:
        summary["latency"] = measure_latency(args, model, splits.test, device, policies, out_dir, log)

    write_csv(out_dir / "methods_test.csv", methods)
    write_json(out_dir / "summary.json", summary)
    log.info(f"Results written to {out_dir}")
    return summary


def evaluate_corruptions(args, model, cfg, data_dir, device, costs, policies, clean_probs, clean_y, out_dir, log):
    """Clean vs corrupted metrics using the thresholds chosen on clean validation data."""
    es = model.exit_stages
    sets: list[tuple[str, str, int, Dataset | None]] = [("clean", "none", 0, None)]
    if args.corruptions == "synthetic":
        for c in SYNTHETIC_CORRUPTIONS:
            for sev in args.severities:
                sets.append(("synthetic", c, sev, build_synthetic_corruption(cfg.dataset, data_dir, c, sev)))
    else:
        if cfg.dataset != "cifar10":
            raise ValueError("CIFAR-10-C can only be used with CIFAR-10 models")
        for c in args.cifar10c_corruptions:
            for sev in args.severities:
                sets.append(("cifar10c", c, sev, CIFAR10C(data_dir, c, sev)))

    rows = []
    for source, corr, sev, ds in sets:
        if ds is None:
            probs, y = clean_probs, clean_y
        else:
            probs, y = collect_probs(model, make_loader(ds, args.batch_size, False, args.num_workers, device), device)
        ems = exit_metrics(probs, y, DATASETS[cfg.dataset].classes)
        base = {"source": source, "corruption": corr, "severity": sev}
        offset = 3 - model.num_exits
        for i, m in enumerate(ems):
            rows.append({**base, "method": f"exit{i + 1 + offset}", "accuracy": m["accuracy"], "ece": m["ece"],
                         "relative_macs": costs.fixed_exit_macs(i, es) / costs.full_macs})
        msg = f"[{source}] {corr} sev {sev}: final-exit acc={ems[-1]['accuracy']:.4f}"
        for name, t in policies.items():
            s = adaptive_summary(probs, y, t, costs, es)
            rows.append({**base, "method": name, "threshold": t, "accuracy": s["accuracy"], "ece": s["ece"],
                         "relative_macs": s["relative_macs"], "exit1_fraction": s["exit1_fraction"],
                         "exit2_fraction": s["exit2_fraction"], "exit3_fraction": s["exit3_fraction"],
                         "early_exit_error_rate": s["early_exit_error_rate"]})
            msg += f" | {name} acc={s['accuracy']:.4f} rel MACs={s['relative_macs']:.3f}"
        log.info(msg)
    write_csv(out_dir / "corruption.csv", rows)
    return {"source": args.corruptions, "severities": args.severities, "rows": len(rows)}


def measure_latency(args, model, test_set, device, policies, out_dir, log):
    n = min(args.latency_samples, len(test_set))
    images = torch.stack([test_set[i][0] for i in range(n)]).to(device)
    modes = {}
    thresholds: dict[str, float] = {}
    if model.num_exits == 1:
        modes["standard_cnn"] = lambda x: fixed_exit_predict(model, x, 0)
    else:
        for e in range(model.num_exits):
            modes[f"multiexit_exit{e + 1}_only"] = lambda x, e=e: fixed_exit_predict(model, x, e)
        thresholds = dict(policies)
        for t in args.latency_thresholds:
            thresholds.setdefault(f"adaptive_t{t:g}", t)
        for name, t in thresholds.items():
            modes[name] = lambda x, t=t: adaptive_predict(model, x, t)
    rows = benchmark_modes(modes, images, args.latency_batch_sizes, args.latency_warmup, args.latency_repeats)
    for r in rows:
        if r["mode"] in thresholds:
            r["threshold"] = thresholds[r["mode"]]
        log.info(f"[latency {r['device']} bs={r['batch_size']}] {r['mode']}: median batch "
                 f"{r['batch_latency_ms_median']:.3f} ms (mean {r['batch_latency_ms_mean']:.3f} "
                 f"+/- {r['batch_latency_ms_std']:.3f}), per-sample {r['per_sample_latency_ms_median']:.4f} ms, "
                 f"{r['throughput_samples_per_s']:.0f} img/s"
                 + (f", peak mem {r['peak_gpu_memory_mb']:.0f} MiB" if "peak_gpu_memory_mb" in r else ""))
    write_csv(out_dir / "latency.csv", rows)
    write_json(out_dir / "latency.json", rows)
    return {"device": str(device), "num_images": n, "batch_sizes": args.latency_batch_sizes,
            "repeats": args.latency_repeats, "warmup": args.latency_warmup}


def main(argv: Sequence[str] | None = None) -> None:
    evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
