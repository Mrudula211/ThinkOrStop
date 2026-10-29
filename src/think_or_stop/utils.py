"""Device selection, seeding, checkpointing and small I/O helpers."""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs).

    With ``deterministic=False`` cuDNN autotuning is enabled for speed, so GPU
    runs with the same seed are statistically but not bitwise identical.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str, strict: bool = True) -> torch.device:
    """Turn a device string into a ``torch.device``, validating CUDA availability.

    If CUDA is requested but unavailable, raise (``strict=True``) or warn and
    fall back to CPU (``strict=False``). We never silently pretend to use a GPU.
    """
    requested = requested.lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            msg = (
                f"CUDA was requested ({requested}) but torch.cuda.is_available() is False. "
                f"PyTorch {torch.__version__} was built with CUDA {torch.version.cuda}. "
                "Install a CUDA-enabled PyTorch build (see README) or pass --device cpu."
            )
            if strict:
                raise RuntimeError(msg)
            warnings.warn(msg + " Falling back to CPU.", stacklevel=2)
            return torch.device("cpu")
        device = torch.device(requested)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"{requested} requested but only {torch.cuda.device_count()} GPU(s) visible")
        return device
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device {requested!r}; use cuda, cuda:N or cpu")


def environment_report(device: torch.device) -> dict[str, Any]:
    """Collect versions and the device actually in use (for logs and results)."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["gpu_name"] = props.name
        info["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
        info["gpu_capability"] = f"{props.major}.{props.minor}"
    return info


def gpu_memory_mb(device: torch.device) -> dict[str, float] | None:
    """Current and peak allocated GPU memory in MiB, or None on CPU."""
    if device.type != "cuda":
        return None
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
    }


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """A logger that writes to stdout and, optionally, to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


def save_checkpoint(path: Path, **state: Any) -> None:
    """Atomically save a checkpoint dict (write to a temp file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False because we store the config dict and RNG states;
    # only load checkpoints you produced yourself.
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------
# JSON / CSV
# --------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return str(obj)  # JSON has no NaN/inf
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(data), fh, indent=2)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header: list[str] = []
    for r in rows:
        header.extend(k for k in r if k not in header)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(_to_jsonable(r) for r in rows)
