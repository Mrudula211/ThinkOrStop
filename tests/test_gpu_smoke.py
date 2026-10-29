import pytest
import torch

from think_or_stop.gpu_check import smoke_step
from think_or_stop.inference import adaptive_predict
from think_or_stop.models import build_model
from think_or_stop.utils import resolve_device

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_cpu_smoke_step():
    r = smoke_step(torch.device("cpu"), "cifar10", "multiexit", batch_size=8)
    assert r["param_device"] == "cpu" and not r["params_without_grad"]


def test_resolve_device_cpu_and_errors():
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError):
        resolve_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")


@cuda
@pytest.mark.cuda
@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
def test_cuda_forward_backward(dataset):
    dev = resolve_device("cuda")
    r = smoke_step(dev, dataset, "multiexit", batch_size=64)
    assert r["param_device"].startswith("cuda") and r["input_device"].startswith("cuda")
    assert r["output_device"].startswith("cuda")
    assert not r["params_without_grad"]


@cuda
@pytest.mark.cuda
def test_cuda_adaptive_matches_cpu():
    torch.manual_seed(0)
    model = build_model(3, "multiexit", 10, base_channels=16).eval()
    x = torch.randn(64, 3, 32, 32)
    cpu = adaptive_predict(model, x, 0.15)
    gpu = adaptive_predict(model.cuda(), x.cuda(), 0.15)
    assert gpu.predictions.is_cuda
    agree = (gpu.predictions.cpu() == cpu.predictions).float().mean().item()
    assert agree >= 0.95  # identical up to rare float differences at the threshold boundary
