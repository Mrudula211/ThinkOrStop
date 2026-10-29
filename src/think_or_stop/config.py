"""Centralised configuration: dataset constants, training defaults and CLI parsing.

Every tunable value lives in :class:`TrainConfig`. Command-line flags are
generated from it, and a JSON file passed with ``--config`` can override the
defaults (explicit CLI flags still win over the JSON file).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

# --------------------------------------------------------------------------
# Dataset constants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Static facts about a supported dataset."""

    name: str
    in_channels: int
    image_size: int
    num_classes: int
    train_size: int  # size of the official training set
    val_size: int  # held out from the official training set
    mean: tuple[float, ...]
    std: tuple[float, ...]
    classes: tuple[str, ...]


# Normalisation constants are the per-channel mean/std of the *official*
# training sets (all 60k MNIST / 50k CIFAR-10 training images). They are
# widely used constants; because they include the images we later hold out
# for validation, the validation split is not perfectly "unseen" with respect
# to these two statistics. The effect is negligible and no test data is used.
DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="mnist",
        in_channels=1,
        image_size=28,
        num_classes=10,
        train_size=60_000,
        val_size=5_000,
        mean=(0.1307,),
        std=(0.3081,),
        classes=tuple(str(i) for i in range(10)),
    ),
    "cifar10": DatasetSpec(
        name="cifar10",
        in_channels=3,
        image_size=32,
        num_classes=10,
        train_size=50_000,
        val_size=5_000,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        classes=(
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ),
    ),
}

MODEL_TYPES = ("baseline", "multiexit")
NUM_STAGES = 3

# --------------------------------------------------------------------------
# Training configuration
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All settings for one training run. Field docs become ``--help`` text."""

    dataset: str = field(default="mnist", metadata={"help": "mnist or cifar10", "choices": tuple(DATASETS)})
    model: str = field(
        default="multiexit",
        metadata={"help": "baseline = final classifier only; multiexit = 3 exits", "choices": MODEL_TYPES},
    )
    epochs: int = field(default=10, metadata={"help": "number of training epochs"})
    batch_size: int = field(default=128, metadata={"help": "mini-batch size (use 64 if VRAM is short)"})
    lr: float = field(default=1e-3, metadata={"help": "initial learning rate"})
    optimizer: str = field(default="adam", metadata={"help": "adam or sgd (SGD uses momentum)", "choices": ("adam", "sgd")})
    momentum: float = field(default=0.9, metadata={"help": "SGD momentum"})
    weight_decay: float = field(default=1e-4, metadata={"help": "L2 weight decay passed to the optimiser"})
    scheduler: str = field(
        default="cosine",
        metadata={"help": "learning-rate schedule", "choices": ("none", "cosine", "step")},
    )
    step_size: int = field(default=10, metadata={"help": "epochs between LR decays for --scheduler step"})
    gamma: float = field(default=0.1, metadata={"help": "LR decay factor for --scheduler step"})
    exit_weights: tuple[float, ...] = field(
        default=(1.0, 1.0, 1.0),
        metadata={"help": "loss weights w1 w2 w3 for exits 1..3 (multiexit only)", "nargs": NUM_STAGES},
    )
    base_channels: int = field(default=64, metadata={"help": "channels in stage 1; stages 2/3 use 2x and 4x"})
    dropout: float = field(default=0.3, metadata={"help": "dropout probability inside each exit head"})
    select_metric: str = field(
        default="final_acc",
        metadata={
            "help": "validation criterion for best.pt: final_acc (exit-3 accuracy) or mean_exit_acc",
            "choices": ("final_acc", "mean_exit_acc"),
        },
    )
    amp: bool = field(default=False, metadata={"help": "use automatic mixed precision (CUDA only)"})
    seed: int = field(default=0, metadata={"help": "seed for weights, shuffling and augmentation"})
    split_seed: int = field(
        default=42,
        metadata={"help": "seed for the train/validation split (kept fixed across --seed values)"},
    )
    deterministic: bool = field(
        default=False,
        metadata={"help": "force deterministic cuDNN kernels (slower, bitwise-reproducible)"},
    )
    num_workers: int = field(default=4, metadata={"help": "DataLoader worker processes"})
    device: str = field(default="cuda", metadata={"help": "cuda, cuda:N or cpu"})
    data_dir: str = field(default="data/raw", metadata={"help": "dataset root directory"})
    output_dir: str = field(default=".", metadata={"help": "root for checkpoints/ and results/"})
    run_name: str = field(default="", metadata={"help": "run identifier (default: <dataset>_<model>_s<seed>)"})
    download: bool = field(default=False, metadata={"help": "download the dataset if it is missing"})
    resume: str = field(default="", metadata={"help": "path to a checkpoint to resume training from"})
    max_train_batches: int = field(
        default=0, metadata={"help": "stop each epoch after N batches (0 = full epoch; for smoke tests)"}
    )

    def __post_init__(self) -> None:
        self.exit_weights = tuple(float(w) for w in self.exit_weights)
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {self.dataset!r}; choose from {sorted(DATASETS)}")
        if self.model not in MODEL_TYPES:
            raise ValueError(f"Unknown model {self.model!r}; choose from {MODEL_TYPES}")
        if len(self.exit_weights) != NUM_STAGES:
            raise ValueError(f"--exit-weights needs {NUM_STAGES} values, got {len(self.exit_weights)}")
        if any(w < 0 for w in self.exit_weights) or sum(self.exit_weights) == 0:
            raise ValueError("--exit-weights must be non-negative and not all zero")
        if self.model == "multiexit" and any(w == 0 for w in self.exit_weights):
            # A zero weight means that head never receives a gradient.
            raise ValueError("multiexit training needs a positive weight for every exit")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("--epochs and --batch-size must be positive")
        if not self.run_name:
            self.run_name = f"{self.dataset}_{self.model}_s{self.seed}"

    # Derived paths -------------------------------------------------------
    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / "checkpoints" / self.run_name

    @property
    def log_dir(self) -> Path:
        return Path(self.output_dir) / "results" / "logs" / self.run_name

    @property
    def spec(self) -> DatasetSpec:
        return DATASETS[self.dataset]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def build_train_parser() -> argparse.ArgumentParser:
    """Build an argparse parser whose flags mirror :class:`TrainConfig`."""
    parser = argparse.ArgumentParser(
        prog="python -m think_or_stop.train",
        description="Train a standard or multi-exit CNN from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="JSON file with default values (CLI flags override it)")
    for f in fields(TrainConfig):
        meta = dict(f.metadata)
        kwargs: dict[str, Any] = {"default": f.default, "help": meta.pop("help", "")}
        if isinstance(f.default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        elif isinstance(f.default, tuple):
            kwargs["type"] = float
            kwargs["nargs"] = meta.pop("nargs", "+")
        else:
            kwargs["type"] = type(f.default)
        if "choices" in meta:
            kwargs["choices"] = meta.pop("choices")
        parser.add_argument(_flag(f.name), dest=f.name, **kwargs)
    return parser


def parse_train_args(argv: Sequence[str] | None = None) -> TrainConfig:
    """Parse CLI arguments (and an optional JSON config) into a TrainConfig."""
    parser = build_train_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        with open(pre.config, encoding="utf-8") as fh:
            overrides = json.load(fh)
        known = {f.name for f in fields(TrainConfig)}
        unknown = set(overrides) - known
        if unknown:
            parser.error(f"{pre.config}: unknown keys {sorted(unknown)}")
        parser.set_defaults(**overrides)
    args = vars(parser.parse_args(argv))
    args.pop("config")
    try:
        return TrainConfig(**args)
    except ValueError as exc:
        parser.error(str(exc))
        raise  # unreachable; parser.error exits
