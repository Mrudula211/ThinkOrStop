"""Train the standard CNN baseline or the multi-exit CNN from scratch.

Example::

    python -m think_or_stop.train --dataset cifar10 --model multiexit --device cuda --epochs 30

Outputs (``<run>`` defaults to ``<dataset>_<model>_s<seed>``):

* ``checkpoints/<run>/best.pt`` - best validation ``--select-metric``
* ``checkpoints/<run>/last.pt`` - end of the latest epoch (for ``--resume``)
* ``results/logs/<run>/{train.log, history.csv, history.json, config.json}``
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig, parse_train_args
from .datasets import build_splits, make_loader
from .losses import MultiExitLoss, build_loss
from .models import ThinkOrStopCNN, build_model, compute_exit_costs, count_parameters
from .utils import (
    environment_report,
    get_logger,
    gpu_memory_mb,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
    write_csv,
    write_json,
)


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, nesterov=True
    )


def build_scheduler(cfg: TrainConfig, opt: torch.optim.Optimizer):
    """Cosine annealing to 0 over the run (default; standard for a fixed epoch
    budget), step decay, or a constant learning rate."""
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    return None


def _autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def train_one_epoch(
    model: ThinkOrStopCNN,
    loader: DataLoader,
    loss_fn: MultiExitLoss,
    opt: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
    max_batches: int = 0,
) -> dict[str, float]:
    """One pass of forward propagation, weighted loss, backpropagation and update."""
    model.train()
    n_exits = model.num_exits
    total_loss, seen = 0.0, 0
    exit_loss = torch.zeros(n_exits, device=device)
    exit_correct = torch.zeros(n_exits, device=device)
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(x)
        loss, per_exit = loss_fn(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        bs = y.size(0)
        seen += bs
        total_loss += loss.detach() * bs
        exit_loss += torch.stack(per_exit) * bs
        exit_correct += torch.stack([(z.argmax(1) == y).sum() for z in logits]).float()
    if seen == 0:
        raise RuntimeError("training loader yielded no batches")
    return _epoch_stats("train", float(total_loss) / seen, (exit_loss / seen).tolist(), (exit_correct / seen).tolist())


@torch.inference_mode()
def validate(
    model: ThinkOrStopCNN, loader: DataLoader, loss_fn: MultiExitLoss, device: torch.device
) -> dict[str, float]:
    """Loss and accuracy of every exit on a held-out loader (no gradients)."""
    model.eval()
    n_exits = model.num_exits
    total_loss, seen = 0.0, 0
    exit_loss = torch.zeros(n_exits, device=device)
    exit_correct = torch.zeros(n_exits, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss, per_exit = loss_fn(logits, y)
        bs = y.size(0)
        seen += bs
        total_loss += float(loss) * bs
        exit_loss += torch.stack(per_exit) * bs
        exit_correct += torch.stack([(z.argmax(1) == y).sum() for z in logits]).float()
    return _epoch_stats("val", total_loss / seen, (exit_loss / seen).tolist(), (exit_correct / seen).tolist())


def _epoch_stats(prefix: str, loss: float, exit_losses: list[float], exit_accs: list[float]) -> dict[str, float]:
    out = {f"{prefix}_loss": loss}
    # Exit numbering: the baseline's single exit is the final one (exit 3).
    offset = 3 - len(exit_accs)
    for i, (l, a) in enumerate(zip(exit_losses, exit_accs)):
        out[f"{prefix}_loss_exit{i + 1 + offset}"] = l
        out[f"{prefix}_acc_exit{i + 1 + offset}"] = a
    out[f"{prefix}_acc_final"] = exit_accs[-1]
    out[f"{prefix}_acc_mean"] = sum(exit_accs) / len(exit_accs)
    return out


def selection_score(cfg: TrainConfig, val_stats: dict[str, float]) -> float:
    return val_stats["val_acc_final"] if cfg.select_metric == "final_acc" else val_stats["val_acc_mean"]


def run(cfg: TrainConfig) -> dict:
    """Full training run. Returns a summary dict (also written to disk)."""
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("think_or_stop.train", cfg.log_dir / "train.log")
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, cfg.deterministic)
    env = environment_report(device)
    log.info("Environment: " + ", ".join(f"{k}={v}" for k, v in env.items()))
    if cfg.amp and device.type != "cuda":
        log.info("--amp ignored: mixed precision is only enabled on CUDA")
        cfg.amp = False

    splits = build_splits(cfg.dataset, cfg.data_dir, cfg.split_seed, download=cfg.download)
    log.info(f"Data: train={len(splits.train)} val={len(splits.val)} test={len(splits.test)} (test unused in training)")
    train_loader = make_loader(splits.train, cfg.batch_size, True, cfg.num_workers, device, seed=cfg.seed)
    val_loader = make_loader(splits.val, cfg.batch_size, False, cfg.num_workers, device)

    spec = cfg.spec
    model = build_model(spec.in_channels, cfg.model, spec.num_classes, cfg.base_channels, cfg.dropout).to(device)
    costs = compute_exit_costs(model, (spec.in_channels, spec.image_size, spec.image_size))
    log.info(
        f"Model: {cfg.model}, exits={model.num_exits}, params={count_parameters(model):,}, "
        f"stage MACs={costs.stage_macs}, head MACs={costs.head_macs}, full MACs={costs.full_macs:,}"
    )
    loss_fn = build_loss(cfg.model, cfg.exit_weights)
    log.info(f"Loss weights per exit: {loss_fn.weights}")
    opt = build_optimizer(cfg, model)
    sched = build_scheduler(cfg, opt)
    scaler = torch.amp.GradScaler("cuda") if cfg.amp else None

    start_epoch, best_score, history = 1, -1.0, []
    if cfg.resume:
        # Load on CPU: RNG states must stay CPU tensors; load_state_dict moves weights/optimizer state.
        ckpt = load_checkpoint(cfg.resume, map_location="cpu")
        if ckpt["config"]["model"] != cfg.model or ckpt["config"]["dataset"] != cfg.dataset:
            raise ValueError("resume checkpoint has a different model/dataset than the current config")
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        if sched is not None and ckpt.get("scheduler_state"):
            sched.load_state_dict(ckpt["scheduler_state"])
            if isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR) and sched.T_max != cfg.epochs:
                # Run extended/shortened on resume: re-target the schedule and reset the LR
                # from the closed-form cosine (the chainable form cannot recover from lr=0).
                sched.T_max = cfg.epochs
                t = sched.last_epoch
                for g, base in zip(opt.param_groups, sched.base_lrs):
                    g["lr"] = sched.eta_min + (base - sched.eta_min) * (1 + math.cos(math.pi * t / cfg.epochs)) / 2
        if scaler is not None and ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        if device.type == "cuda" and ckpt["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt["best_score"]
        history = ckpt["history"]
        # Re-seed the shuffle so a resumed run does not replay epoch-1 order.
        train_loader.generator.manual_seed(cfg.seed + ckpt["epoch"])  # type: ignore[union-attr]
        log.info(f"Resumed from {cfg.resume} at epoch {start_epoch} (best {cfg.select_metric}={best_score:.4f})")

    write_json(cfg.log_dir / "config.json", {"config": cfg.to_dict(), "environment": env})

    # Verify that parameters and batches really live on the requested device.
    xb, _ = next(iter(train_loader))
    xb = xb.to(device, non_blocking=True)
    log.info(f"Device check: model params on {next(model.parameters()).device}, first batch on {xb.device}, shape {tuple(xb.shape)}")
    del xb

    for epoch in range(start_epoch, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        lr = opt.param_groups[0]["lr"]
        tr = train_one_epoch(model, train_loader, loss_fn, opt, device, scaler, cfg.amp, cfg.max_train_batches)
        va = validate(model, val_loader, loss_fn, device)
        if sched is not None:
            sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        mem = gpu_memory_mb(device)
        row = {"epoch": epoch, "lr": lr, "epoch_seconds": elapsed, **tr, **va}
        if mem:
            row["gpu_peak_allocated_mb"] = mem["peak_allocated_mb"]
        history.append(row)

        score = selection_score(cfg, va)
        improved = score > best_score
        if improved:
            best_score = score
        exit_accs = " ".join(f"e{k[-1]}={v:.4f}" for k, v in va.items() if k.startswith("val_acc_exit"))
        log.info(
            f"epoch {epoch:3d}/{cfg.epochs} | {elapsed:6.1f}s | lr {lr:.2e} | train loss {tr['train_loss']:.4f} "
            f"acc {tr['train_acc_final']:.4f} | val loss {va['val_loss']:.4f} acc[{exit_accs}]"
            + (f" | gpu peak {mem['peak_allocated_mb']:.0f} MiB" if mem else "")
            + (" | *best*" if improved else "")
        )

        state = dict(
            config=cfg.to_dict(),
            model_state=model.state_dict(),
            optimizer_state=opt.state_dict(),
            scheduler_state=sched.state_dict() if sched is not None else None,
            scaler_state=scaler.state_dict() if scaler is not None else None,
            epoch=epoch,
            best_score=best_score,
            select_metric=cfg.select_metric,
            history=history,
            rng={"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
            exit_costs={"stage_macs": costs.stage_macs, "head_macs": costs.head_macs},
        )
        save_checkpoint(cfg.checkpoint_dir / "last.pt", **state)
        if improved:
            save_checkpoint(cfg.checkpoint_dir / "best.pt", **state)
        write_csv(cfg.log_dir / "history.csv", history)
        write_json(cfg.log_dir / "history.json", history)

    summary = {
        "run_name": cfg.run_name,
        "best_checkpoint": str(cfg.checkpoint_dir / "best.pt"),
        "last_checkpoint": str(cfg.checkpoint_dir / "last.pt"),
        "select_metric": cfg.select_metric,
        "best_score": best_score,
        "epochs_completed": len(history),
        "total_train_seconds": sum(r["epoch_seconds"] for r in history),
    }
    write_json(cfg.log_dir / "summary.json", summary)
    log.info(f"Done. best {cfg.select_metric}={best_score:.4f}; checkpoint {summary['best_checkpoint']}")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_train_args(argv))


if __name__ == "__main__":
    main()
