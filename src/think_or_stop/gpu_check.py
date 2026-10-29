"""Verify the GPU and run a one-batch forward/backward smoke test.

    python -m think_or_stop.gpu_check --device cuda

Uses random tensors, so it needs no dataset. Exits with status 1 if the
requested device cannot be used.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Sequence

import torch

from .config import DATASETS
from .losses import build_loss
from .models import build_model
from .utils import environment_report, resolve_device


def smoke_step(device: torch.device, dataset: str, model_type: str, batch_size: int = 128) -> dict:
    """One optimisation step on random data; returns loss, timings and devices."""
    spec = DATASETS[dataset]
    model = build_model(spec.in_channels, model_type, spec.num_classes).to(device)
    loss_fn = build_loss(model_type, (1.0, 1.0, 1.0))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch_size, spec.in_channels, spec.image_size, spec.image_size, device=device)
    y = torch.randint(0, spec.num_classes, (batch_size,), device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    logits = model(x)
    loss, _ = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    return {
        "dataset": dataset,
        "model": model_type,
        "loss": float(loss.detach()),
        "seconds": time.perf_counter() - t0,
        "param_device": str(next(model.parameters()).device),
        "input_device": str(x.device),
        "output_device": str(logits[-1].device),
        "exit_shapes": [tuple(z.shape) for z in logits],
        "params_without_grad": missing,
        "peak_mem_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
    }


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m think_or_stop.gpu_check", description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args(argv)
    try:
        device = resolve_device(args.device)
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)
    for k, v in environment_report(device).items():
        print(f"{k:>22}: {v}")
    ok = True
    for ds in ("mnist", "cifar10"):
        for mt in ("baseline", "multiexit"):
            r = smoke_step(device, ds, mt, args.batch_size)
            good = r["param_device"].startswith(device.type) and not r["params_without_grad"]
            ok &= good
            print(f"[{'ok' if good else 'FAIL'}] {ds:7s} {mt:9s} loss={r['loss']:.3f} exits={r['exit_shapes']} "
                  f"params/input/output on {r['param_device']}/{r['input_device']}/{r['output_device']} "
                  f"step={r['seconds'] * 1e3:.1f} ms" + (f" peak={r['peak_mem_mb']:.0f} MiB" if r["peak_mem_mb"] else ""))
            if r["params_without_grad"]:
                print("   parameters without gradient:", r["params_without_grad"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
