import pytest
import torch

from think_or_stop.config import DATASETS
from think_or_stop.models import build_model, compute_exit_costs, count_parameters


@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
@pytest.mark.parametrize("model_type,n_exits", [("baseline", 1), ("multiexit", 3)])
def test_output_shapes(dataset, model_type, n_exits):
    spec = DATASETS[dataset]
    model = build_model(spec.in_channels, model_type, spec.num_classes, base_channels=16)
    x = torch.randn(5, spec.in_channels, spec.image_size, spec.image_size)
    out = model(x)
    assert len(out) == n_exits == model.num_exits
    for z in out:
        assert z.shape == (5, spec.num_classes)
        assert torch.isfinite(z).all()


@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
def test_fixed_exit_matches_full_forward(dataset):
    spec = DATASETS[dataset]
    model = build_model(spec.in_channels, "multiexit", spec.num_classes, base_channels=16).eval()
    x = torch.randn(4, spec.in_channels, spec.image_size, spec.image_size)
    full = model(x)
    for e in range(3):
        torch.testing.assert_close(model.forward_to_exit(x, e), full[e])


def test_wrong_channel_count_fails():
    model = build_model(1, "multiexit", 10, base_channels=8)
    with pytest.raises(RuntimeError):
        model(torch.randn(2, 3, 28, 28))


def test_baseline_shares_backbone_architecture():
    b = build_model(3, "baseline", 10)
    m = build_model(3, "multiexit", 10)
    assert count_parameters(b.stages) == count_parameters(m.stages)
    assert count_parameters(b.heads[0]) == count_parameters(m.heads[2])
    assert b.exit_stages == (2,) and m.exit_stages == (0, 1, 2)


def test_mac_counts():
    m = build_model(3, "multiexit", 10, base_channels=64)
    c = compute_exit_costs(m, (3, 32, 32))
    # Stage 1: conv 3->64 and 64->64 at 32x32, 3x3 kernels.
    assert c.stage_macs[0] == 64 * 32 * 32 * 3 * 9 + 64 * 32 * 32 * 64 * 9
    # Head 1: Linear(64*4, 128) + Linear(128, 10).
    assert c.head_macs[0] == 64 * 4 * 128 + 128 * 10
    assert c.fixed_exit_macs(0, m.exit_stages) < c.fixed_exit_macs(1, m.exit_stages) < c.full_macs
    assert c.adaptive_exit_macs(2, m.exit_stages) == sum(c.stage_macs) + sum(c.head_macs)
    b = build_model(3, "baseline", 10, base_channels=64)
    assert compute_exit_costs(b, (3, 32, 32)).full_macs == c.full_macs
