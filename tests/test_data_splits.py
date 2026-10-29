import numpy as np
import pytest
import torch

from think_or_stop.datasets import SyntheticCorruption, build_splits, split_indices
from think_or_stop.config import DATASETS

from .conftest import DATA_DIR, has_dataset


def test_split_reproducible_and_disjoint():
    a_tr, a_va = split_indices(50_000, 5_000, seed=42)
    b_tr, b_va = split_indices(50_000, 5_000, seed=42)
    np.testing.assert_array_equal(a_tr, b_tr)
    np.testing.assert_array_equal(a_va, b_va)
    assert len(a_va) == 5_000 and len(a_tr) == 45_000
    assert len(np.intersect1d(a_tr, a_va)) == 0
    assert len(np.union1d(a_tr, a_va)) == 50_000
    _, c_va = split_indices(50_000, 5_000, seed=7)
    assert not np.array_equal(a_va, c_va)


def test_split_rejects_bad_sizes():
    with pytest.raises(ValueError):
        split_indices(10, 10, 0)


@pytest.mark.data
@pytest.mark.skipif(not has_dataset("mnist"), reason="MNIST not downloaded")
def test_mnist_splits():
    s = build_splits("mnist", DATA_DIR, split_seed=42)
    assert (len(s.train), len(s.val), len(s.test)) == (55_000, 5_000, 10_000)
    x, y = s.val[0]
    assert x.shape == (1, 28, 28) and 0 <= y < 10
    # Validation images are deterministic (no augmentation).
    assert torch.equal(s.val[3][0], s.val[3][0])
    s2 = build_splits("mnist", DATA_DIR, split_seed=42)
    np.testing.assert_array_equal(s.val_indices, s2.val_indices)


@pytest.mark.data
@pytest.mark.skipif(not has_dataset("cifar10"), reason="CIFAR-10 not downloaded")
def test_cifar10_splits_and_augmentation():
    s = build_splits("cifar10", DATA_DIR, split_seed=42)
    assert (len(s.train), len(s.val), len(s.test)) == (45_000, 5_000, 10_000)
    assert s.val[0][0].shape == (3, 32, 32)
    assert torch.equal(s.val[5][0], s.val[5][0])
    assert torch.equal(s.test[5][0], s.test[5][0])
    torch.manual_seed(0)
    draws = [s.train[0][0] for _ in range(8)]
    assert any(not torch.equal(draws[0], d) for d in draws[1:])  # random crop/flip on train only


def test_synthetic_corruption_is_deterministic():
    spec = DATASETS["cifar10"]
    base = [(torch.rand(3, 32, 32), 1) for _ in range(4)]
    a = SyntheticCorruption(base, spec, "gaussian_noise", 3, seed=1)
    b = SyntheticCorruption(base, spec, "gaussian_noise", 3, seed=1)
    assert torch.equal(a[2][0], b[2][0])
    assert not torch.equal(a[1][0], a[2][0])
    blur = SyntheticCorruption(base, spec, "gaussian_blur", 5)
    assert blur[0][0].shape == (3, 32, 32)
