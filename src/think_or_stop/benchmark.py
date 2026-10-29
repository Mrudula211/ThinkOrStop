"""Wall-clock latency and throughput measurement.

Protocol
--------
* ``model.eval()`` and ``torch.inference_mode()``; float32; cuDNN autotuning
  disabled so dynamic (shrinking) batch shapes in adaptive mode are not
  penalised by one-off tuning and every mode runs under the same setting.
* A fixed set of real test images is pre-loaded on the device, so data
  loading and host-to-device copies are *excluded* from the timings.
* ``warmup`` untimed batches run before measuring each mode.
* Each batch is timed individually with ``time.perf_counter`` and, on CUDA,
  ``torch.cuda.synchronize()`` immediately before and after the call.
* The whole image set is processed ``repeats`` times; we report batch latency
  statistics (mean / median / std / p90), the *amortised* per-sample latency
  (batch latency / batch size; equal to true per-sample latency only when
  batch size is 1), throughput, and the spread of throughput across repeats.
* Peak GPU memory is ``torch.cuda.max_memory_allocated`` during the mode.

Lower MACs do not guarantee lower latency: adaptive inference adds softmax,
boolean-mask indexing (a GPU->CPU sync) and smaller, less efficient kernels.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def time_mode(
    fn: Callable[[torch.Tensor], object],
    images: torch.Tensor,
    batch_size: int,
    warmup: int = 20,
    repeats: int = 5,
) -> dict[str, float]:
    """Time ``fn`` over ``images`` (already on the target device) in batches."""
    device = images.device
    batches = list(torch.split(images, batch_size))
    for i in range(warmup):
        fn(batches[i % len(batches)])
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    lat_ms: list[float] = []
    sizes: list[int] = []
    round_throughput: list[float] = []
    for _ in range(repeats):
        round_time = 0.0
        for xb in batches:
            _sync(device)
            t0 = time.perf_counter()
            fn(xb)
            _sync(device)
            dt = time.perf_counter() - t0
            round_time += dt
            lat_ms.append(dt * 1e3)
            sizes.append(xb.size(0))
        round_throughput.append(images.size(0) / round_time)

    lat = np.array(lat_ms)
    per_sample = lat / np.array(sizes)
    out = {
        "batch_size": batch_size,
        "num_images": int(images.size(0)),
        "repeats": repeats,
        "batch_latency_ms_mean": float(lat.mean()),
        "batch_latency_ms_median": float(np.median(lat)),
        "batch_latency_ms_std": float(lat.std(ddof=1)) if len(lat) > 1 else 0.0,
        "batch_latency_ms_p90": float(np.percentile(lat, 90)),
        "per_sample_latency_ms_mean": float(per_sample.mean()),
        "per_sample_latency_ms_median": float(np.median(per_sample)),
        "throughput_samples_per_s": float(np.mean(round_throughput)),
        "throughput_std": float(np.std(round_throughput, ddof=1)) if repeats > 1 else 0.0,
    }
    if device.type == "cuda":
        out["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return out


def benchmark_modes(
    modes: dict[str, Callable[[torch.Tensor], object]],
    images: torch.Tensor,
    batch_sizes: list[int],
    warmup: int = 20,
    repeats: int = 5,
) -> list[dict]:
    """Benchmark every named mode at every batch size under identical settings."""
    prev = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        rows = []
        for bs in batch_sizes:
            for name, fn in modes.items():
                r = time_mode(fn, images, bs, warmup, repeats)
                rows.append({"mode": name, "device": str(images.device), **r})
        return rows
    finally:
        torch.backends.cudnn.benchmark = prev
