from pathlib import Path

import pytest
import torch

from think_or_stop.models import build_model

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture
def small_multiexit():
    torch.manual_seed(0)
    return build_model(3, "multiexit", 10, base_channels=8).eval()


@pytest.fixture
def small_baseline():
    torch.manual_seed(0)
    return build_model(3, "baseline", 10, base_channels=8).eval()


def has_dataset(name: str) -> bool:
    if name == "mnist":
        return (DATA_DIR / "MNIST" / "raw" / "t10k-labels-idx1-ubyte").is_file()
    return (DATA_DIR / "cifar-10-batches-py" / "test_batch").is_file()
