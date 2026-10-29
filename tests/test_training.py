"""Loss/backprop, CPU training smoke test, checkpoint round-trip and config parsing."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from think_or_stop.config import TrainConfig, parse_train_args
from think_or_stop.losses import MultiExitLoss, build_loss
from think_or_stop.models import build_model
from think_or_stop.train import build_optimizer, train_one_epoch, validate
from think_or_stop.utils import load_checkpoint, save_checkpoint


def _loader(n=48, c=3, s=32, bs=16):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, c, s, s, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def test_weighted_loss_and_all_heads_get_gradients():
    model = build_model(3, "multiexit", 10, base_channels=8)
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
    logits = model(x)
    loss_fn = MultiExitLoss([0.2, 0.3, 1.0])
    total, per = loss_fn(logits, y)
    assert total.item() == pytest.approx(0.2 * per[0].item() + 0.3 * per[1].item() + per[2].item(), rel=1e-5)
    total.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for head in model.heads:
        assert head[-1].weight.grad.abs().sum() > 0


def test_baseline_loss_is_plain_ce():
    model = build_model(1, "baseline", 10, base_channels=8).eval()  # eval: dropout off, outputs repeatable
    loss_fn = build_loss("baseline", (1, 1, 1))
    x, y = torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,))
    total, _ = loss_fn(model(x), y)
    torch.testing.assert_close(total, torch.nn.functional.cross_entropy(model(x)[0], y))


def test_cpu_training_smoke_reduces_loss():
    torch.manual_seed(0)
    model = build_model(3, "multiexit", 10, base_channels=8)
    cfg = TrainConfig(dataset="cifar10", model="multiexit", lr=3e-3, device="cpu")
    opt = build_optimizer(cfg, model)
    loss_fn = build_loss("multiexit", cfg.exit_weights)
    loader = _loader()
    dev = torch.device("cpu")
    first = train_one_epoch(model, loader, loss_fn, opt, dev)
    for _ in range(15):
        last = train_one_epoch(model, loader, loss_fn, opt, dev)
    assert last["train_loss"] < first["train_loss"]
    v = validate(model, loader, loss_fn, dev)
    assert {"val_acc_exit1", "val_acc_exit2", "val_acc_exit3", "val_acc_final"} <= set(v)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = build_model(3, "multiexit", 10, base_channels=8).eval()
    cfg = TrainConfig(dataset="cifar10", model="multiexit", base_channels=8)
    save_checkpoint(tmp_path / "c.pt", config=cfg.to_dict(), model_state=model.state_dict(), epoch=3)
    ckpt = load_checkpoint(tmp_path / "c.pt")
    cfg2 = TrainConfig.from_dict(ckpt["config"])
    assert cfg2 == cfg
    model2 = build_model(3, "multiexit", 10, base_channels=cfg2.base_channels).eval()
    model2.load_state_dict(ckpt["model_state"])
    x = torch.randn(3, 3, 32, 32)
    for a, b in zip(model(x), model2(x)):
        torch.testing.assert_close(a, b)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")


def test_config_parsing(tmp_path):
    cfg = parse_train_args(["--dataset", "cifar10", "--model", "baseline", "--epochs", "3", "--exit-weights",
                            "0.5", "0.5", "1", "--no-amp", "--optimizer", "sgd"])
    assert cfg.dataset == "cifar10" and cfg.epochs == 3 and cfg.exit_weights == (0.5, 0.5, 1.0)
    assert cfg.optimizer == "sgd" and cfg.run_name == "cifar10_baseline_s0"
    p = tmp_path / "c.json"
    p.write_text('{"dataset": "mnist", "epochs": 7, "lr": 0.01}')
    cfg = parse_train_args(["--config", str(p), "--epochs", "2"])
    assert cfg.dataset == "mnist" and cfg.epochs == 2 and cfg.lr == 0.01
    with pytest.raises(SystemExit):
        parse_train_args(["--exit-weights", "1", "0", "1"])


@pytest.mark.data
def test_train_and_resume_end_to_end_cpu(tmp_path):
    from think_or_stop.train import run

    from .conftest import DATA_DIR, has_dataset

    if not has_dataset("mnist"):
        pytest.skip("MNIST not downloaded")
    common = dict(dataset="mnist", model="multiexit", device="cpu", num_workers=0, batch_size=64,
                  base_channels=8, max_train_batches=3, data_dir=str(DATA_DIR), output_dir=str(tmp_path))
    run(TrainConfig(epochs=1, **common))
    ckpt_dir = tmp_path / "checkpoints" / "mnist_multiexit_s0"
    assert (ckpt_dir / "best.pt").is_file() and (ckpt_dir / "last.pt").is_file()
    summary = run(TrainConfig(epochs=2, resume=str(ckpt_dir / "last.pt"), **common))
    assert summary["epochs_completed"] == 2
    ckpt = load_checkpoint(ckpt_dir / "last.pt")
    assert ckpt["epoch"] == 2 and len(ckpt["history"]) == 2
    assert ckpt["history"][1]["lr"] > 0
