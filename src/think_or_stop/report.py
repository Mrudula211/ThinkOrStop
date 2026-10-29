"""Aggregate evaluation results across runs/seeds into comparison tables.

    python -m think_or_stop.report --metrics-dirs results/metrics/cifar10_baseline_s0 \
        results/metrics/cifar10_multiexit_s0 --out results/metrics/report_cifar10

Rows are grouped by (dataset, method). With several seeds per method the table
shows mean +/- sample standard deviation (n-1); with one seed the std is
blank. This is descriptive only: no significance test is performed, so
differences smaller than the seed-to-seed spread should not be interpreted.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .utils import read_json, write_csv

METRICS = ("accuracy", "ece", "relative_macs", "mean_exit_depth", "early_exit_error_rate")
LATENCY = ("per_sample_latency_ms_median", "throughput_samples_per_s")
METHOD_ORDER = ("standard_cnn", "multiexit_exit3_only", "multiexit_exit2_only", "multiexit_exit1_only")


def _float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def collect(dirs: Sequence[Path]) -> list[dict]:
    """One flat record per (run, method), with latency columns per batch size."""
    records = []
    for d in dirs:
        s = read_json(d / "summary.json")
        lat: dict[str, dict[int, dict]] = defaultdict(dict)
        if (d / "latency.csv").is_file():
            with open(d / "latency.csv", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    lat[r["mode"]][int(r["batch_size"])] = r
        for m in s["methods"]:
            rec = {"dataset": s["dataset"], "run": s["run_name"], "seed": s["seed"], "method": m["method"]}
            rec.update({k: _float(m.get(k)) for k in METRICS})
            rec["threshold"] = _float(m.get("threshold"))
            for bs, r in lat.get(m["method"], {}).items():
                for k in LATENCY:
                    rec[f"{k}_bs{bs}"] = _float(r[k])
            records.append(rec)
    return records


def aggregate(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["dataset"], r["method"])].append(r)
    keys = [k for k in records[0] if k not in ("dataset", "run", "seed", "method")] if records else []
    for r in records:
        keys += [k for k in r if k not in keys and k not in ("dataset", "run", "seed", "method")]
    out = []
    for (ds, method), rs in groups.items():
        row = {"dataset": ds, "method": method, "n_seeds": len(rs), "seeds": " ".join(str(r["seed"]) for r in rs)}
        for k in keys:
            vals = np.array([r.get(k, np.nan) for r in rs], dtype=float)
            vals = vals[~np.isnan(vals)]
            row[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        out.append(row)

    def order(r):
        m = r["method"]
        return (r["dataset"], METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER), m)

    return sorted(out, key=order)


def _fmt(row: dict, key: str, scale: float = 1.0, digits: int = 2) -> str:
    m, s = row.get(f"{key}_mean", float("nan")), row.get(f"{key}_std", float("nan"))
    if np.isnan(m):
        return "-"
    txt = f"{m * scale:.{digits}f}"
    return txt + (f" ± {s * scale:.{digits}f}" if not np.isnan(s) else "")


def to_markdown(rows: list[dict]) -> str:
    bss = sorted({int(k.split("_bs")[1].split("_")[0]) for r in rows for k in r if "_bs" in k})
    head = ["dataset", "method", "seeds", "test acc (%)", "ECE", "MACs (% of full)", "mean exit depth",
            "early-exit error (%)"]
    head += [f"latency bs={b} (ms/img)" for b in bss]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["dataset"], r["method"], str(r["n_seeds"]), _fmt(r, "accuracy", 100), _fmt(r, "ece", 1, 3),
                 _fmt(r, "relative_macs", 100, 1), _fmt(r, "mean_exit_depth", 1, 2),
                 _fmt(r, "early_exit_error_rate", 100)]
        cells += [_fmt(r, f"per_sample_latency_ms_median_bs{b}", 1, 4) for b in bss]
        lines.append("| " + " | ".join(cells) + " |")
    note = ("\nValues are mean ± sample std over seeds (std omitted for a single seed). Latency is the median "
            "per-sample latency (batch latency / batch size). MACs are analytic estimates (Conv2d + Linear only). "
            "No statistical significance testing was performed.\n")
    return "\n".join(lines) + "\n" + note


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m think_or_stop.report", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-dirs", nargs="+", required=True, help="results/metrics/<run> directories")
    p.add_argument("--out", default="results/metrics/report", help="output path prefix (.csv and .md are added)")
    args = p.parse_args(argv)
    dirs = [Path(d) for d in args.metrics_dirs]
    missing = [d for d in dirs if not (d / "summary.json").is_file()]
    if missing:
        p.error(f"no summary.json in: {', '.join(map(str, missing))} (run think_or_stop.evaluate first)")
    records = collect(dirs)
    rows = aggregate(records)
    out = Path(args.out)
    write_csv(out.with_name(out.name + "_runs.csv"), records)
    write_csv(out.with_suffix(".csv"), rows)
    md = to_markdown(rows)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {out.with_suffix('.csv')}, {out.with_suffix('.md')}, {out.with_name(out.name + '_runs.csv')}")


if __name__ == "__main__":
    main()
