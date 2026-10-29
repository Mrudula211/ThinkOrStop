"""Dataset loading, reproducible train/validation splits and corruption datasets.

Splits
------
* ``train`` - official training set minus the validation hold-out; random
  augmentation (CIFAR-10 only: 4-px padded random crop + horizontal flip).
* ``val``   - ``spec.val_size`` images held out from the official training set
  with a fixed ``split_seed``; no augmentation. Used for checkpoint selection
  and threshold selection.
* ``test``  - the official test set; no augmentation. Used only for final
  reporting, never for tuning.

Corruptions
-----------
:class:`SyntheticCorruption` re-implements two corruptions (Gaussian noise and
Gaussian blur) with the severity parameters published for CIFAR-10-C. These
are *synthetic* corruptions generated here and are reported separately from the
real CIFAR-10-C benchmark (:class:`CIFAR10C`), which must be downloaded
explicitly (see :func:`download_cifar10c`).
"""

from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tvd
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from .config import DATASETS, DatasetSpec

# --------------------------------------------------------------------------
# Transforms and official datasets
# --------------------------------------------------------------------------


def build_transforms(spec: DatasetSpec, augment: bool) -> T.Compose:
    """Training (``augment=True``) or evaluation transform for ``spec``."""
    steps: list[Callable] = []
    if augment and spec.name == "cifar10":
        steps += [T.RandomCrop(spec.image_size, padding=4), T.RandomHorizontalFlip()]
    # MNIST: no augmentation (horizontal flips would change digit identity).
    steps += [T.ToTensor(), T.Normalize(spec.mean, spec.std)]
    return T.Compose(steps)


def load_official(
    name: str,
    root: str | Path,
    train: bool,
    transform: Callable | None,
    download: bool = False,
) -> Dataset:
    """Load the official torchvision MNIST/CIFAR-10 split.

    Network access only happens when ``download=True``.
    """
    root = Path(root)
    cls = {"mnist": tvd.MNIST, "cifar10": tvd.CIFAR10}[name]
    try:
        return cls(root=str(root), train=train, transform=transform, download=download)
    except RuntimeError as exc:
        if download:
            raise
        raise RuntimeError(
            f"{name} not found under {root.resolve()}. Download it first with:\n"
            f"  python -m think_or_stop.download --dataset {name} --data-dir {root}\n"
            f"or pass --download to the training command."
        ) from exc


def split_indices(n: int, val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split ``range(n)`` into disjoint train/val index arrays."""
    if not 0 < val_size < n:
        raise ValueError(f"val_size must be in (0, {n}), got {val_size}")
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).numpy()
    val_idx = np.sort(perm[:val_size])
    train_idx = np.sort(perm[val_size:])
    return train_idx, val_idx


@dataclass
class DataSplits:
    train: Dataset
    val: Dataset
    test: Dataset
    train_indices: np.ndarray
    val_indices: np.ndarray


def build_splits(
    dataset: str,
    data_dir: str | Path,
    split_seed: int,
    download: bool = False,
    augment: bool = True,
) -> DataSplits:
    """Return train/val/test datasets with the documented transforms."""
    spec = DATASETS[dataset]
    train_full = load_official(dataset, data_dir, True, build_transforms(spec, augment), download)
    # Second view of the same training files with the deterministic transform.
    val_full = load_official(dataset, data_dir, True, build_transforms(spec, False), False)
    test = load_official(dataset, data_dir, False, build_transforms(spec, False), False)
    if len(train_full) != spec.train_size:
        raise RuntimeError(f"Expected {spec.train_size} training images, found {len(train_full)}")
    train_idx, val_idx = split_indices(len(train_full), spec.val_size, split_seed)
    return DataSplits(
        train=Subset(train_full, train_idx.tolist()),
        val=Subset(val_full, val_idx.tolist()),
        test=test,
        train_indices=train_idx,
        val_indices=val_idx,
    )


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """DataLoader with pinned memory on CUDA and a seeded shuffle generator."""
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        generator=gen,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )


def _seed_worker(worker_id: int) -> None:
    # Make NumPy/random inside workers depend on the torch-assigned worker seed.
    import random

    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# --------------------------------------------------------------------------
# Synthetic corruptions (reproducible, applied before normalisation)
# --------------------------------------------------------------------------

# Severity parameters follow Hendrycks & Dietterich (2019), CIFAR-10-C code:
# gaussian_noise: additive N(0, sigma^2) on [0,1] pixels, then clipped.
# gaussian_blur:  Gaussian blur with standard deviation sigma (pixels).
SYNTHETIC_CORRUPTIONS: dict[str, tuple[float, ...]] = {
    "gaussian_noise": (0.04, 0.06, 0.08, 0.09, 0.10),
    "gaussian_blur": (0.4, 0.6, 0.7, 0.8, 1.0),
}


class SyntheticCorruption(Dataset):
    """Wrap an official test set and apply a deterministic corruption.

    Each image ``i`` gets its own RNG seeded with ``seed + i``, so the noise
    pattern is identical across runs, workers and batch sizes.
    """

    def __init__(self, base: Dataset, spec: DatasetSpec, corruption: str, severity: int, seed: int = 1234):
        if corruption not in SYNTHETIC_CORRUPTIONS:
            raise ValueError(f"Unknown corruption {corruption!r}; choose from {sorted(SYNTHETIC_CORRUPTIONS)}")
        if not 1 <= severity <= 5:
            raise ValueError("severity must be in 1..5")
        self.base = base  # must return un-normalised [0,1] tensors
        self.corruption = corruption
        self.severity = severity
        self.param = SYNTHETIC_CORRUPTIONS[corruption][severity - 1]
        self.seed = seed
        self.normalize = T.Normalize(spec.mean, spec.std)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if self.corruption == "gaussian_noise":
            gen = torch.Generator().manual_seed(self.seed + idx)
            x = (x + torch.randn(x.shape, generator=gen) * self.param).clamp(0.0, 1.0)
        else:  # gaussian_blur
            k = 2 * math.ceil(3 * self.param) + 1
            x = TF.gaussian_blur(x, kernel_size=[k, k], sigma=[self.param, self.param])
        return self.normalize(x), y


def build_synthetic_corruption(
    dataset: str, data_dir: str | Path, corruption: str, severity: int, seed: int = 1234
) -> SyntheticCorruption:
    spec = DATASETS[dataset]
    base = load_official(dataset, data_dir, False, T.ToTensor(), False)
    return SyntheticCorruption(base, spec, corruption, severity, seed)


# --------------------------------------------------------------------------
# CIFAR-10-C (optional; real benchmark)
# --------------------------------------------------------------------------

# Source: Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to
# Common Corruptions and Perturbations", ICLR 2019.
# Official distribution: Zenodo record 2535967, licensed CC BY 4.0.
CIFAR10C_RECORD_API = "https://zenodo.org/api/records/2535967"
CIFAR10C_ARCHIVE = "CIFAR-10-C.tar"
CIFAR10C_DIRNAME = "CIFAR-10-C"
CIFAR10C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
)  # the 15 benchmark corruptions (4 extra "hold-out" ones are also in the archive)


def cifar10c_metadata(api_url: str = CIFAR10C_RECORD_API, timeout: float = 30.0) -> dict:
    """Query Zenodo for the archive's download URL, size and checksum."""
    with urllib.request.urlopen(api_url, timeout=timeout) as resp:
        record = json.load(resp)
    for f in record.get("files", []):
        if f.get("key") == CIFAR10C_ARCHIVE:
            algo, _, digest = f["checksum"].partition(":")
            return {
                "record": record.get("links", {}).get("self_html") or record.get("links", {}).get("html"),
                "doi": record.get("doi"),
                "license": (record.get("metadata", {}).get("license") or {}).get("id"),
                "url": f["links"]["self"],
                "size_bytes": f.get("size"),
                "checksum_algorithm": algo,
                "checksum": digest,
            }
    raise RuntimeError(f"{CIFAR10C_ARCHIVE} not listed in Zenodo record {api_url}")


def download_cifar10c(root: str | Path) -> Path:
    """Download and verify CIFAR-10-C from Zenodo, then extract it under ``root``."""
    from torchvision.datasets.utils import download_url, extract_archive

    root = Path(root)
    target = root / CIFAR10C_DIRNAME
    if (target / "labels.npy").is_file():
        return target
    meta = cifar10c_metadata()
    if meta["checksum_algorithm"] != "md5":
        raise RuntimeError(f"Unexpected checksum algorithm {meta['checksum_algorithm']}")
    download_url(meta["url"], str(root), filename=CIFAR10C_ARCHIVE, md5=meta["checksum"])
    extract_archive(str(root / CIFAR10C_ARCHIVE), str(root))
    meta["downloaded_utc"] = datetime.now(timezone.utc).isoformat()
    (target / "SOURCE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return target


class CIFAR10C(Dataset):
    """One corruption/severity slice of CIFAR-10-C (10,000 test images)."""

    def __init__(self, root: str | Path, corruption: str, severity: int):
        base = Path(root) / CIFAR10C_DIRNAME
        if not (base / "labels.npy").is_file():
            raise FileNotFoundError(
                f"CIFAR-10-C not found in {base.resolve()}. Download it with:\n"
                f"  python -m think_or_stop.download --dataset cifar10c --data-dir {root}"
            )
        if not 1 <= severity <= 5:
            raise ValueError("severity must be in 1..5")
        sl = slice((severity - 1) * 10_000, severity * 10_000)
        self.images = np.load(base / f"{corruption}.npy", mmap_mode="r")[sl]
        self.labels = np.load(base / "labels.npy")[sl].astype(np.int64)
        spec = DATASETS["cifar10"]
        self.transform = T.Compose([T.ToTensor(), T.Normalize(spec.mean, spec.std)])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.transform(np.array(self.images[idx])), int(self.labels[idx])
