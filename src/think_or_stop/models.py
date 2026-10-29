"""Three-stage CNN backbone with optional intermediate (early-exit) classifiers.

Architecture (C = ``base_channels``, default 64)::

    input (1x28x28 MNIST or 3x32x32 CIFAR-10)
    Stage 1: [Conv3x3(C)  - BN - ReLU] x2 - MaxPool2   -> Exit 1 head
    Stage 2: [Conv3x3(2C) - BN - ReLU] x2 - MaxPool2   -> Exit 2 head
    Stage 3: [Conv3x3(4C) - BN - ReLU] x2 - MaxPool2   -> Exit 3 head (final)

    Exit head: AdaptiveAvgPool(2x2) - Flatten - Linear(4c, 128) - ReLU
               - Dropout(p) - Linear(128, num_classes)

The standard CNN baseline (``multi_exit=False``) has exactly the same stages
and the same final head, but no intermediate heads, so the two variants differ
only in the presence of exits 1 and 2 and the training objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import NUM_STAGES

HEAD_HIDDEN = 128
HEAD_POOL = 2


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def make_stage(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(conv_block(in_ch, out_ch), conv_block(out_ch, out_ch), nn.MaxPool2d(2))


def make_head(in_ch: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(HEAD_POOL),
        nn.Flatten(),
        nn.Linear(in_ch * HEAD_POOL * HEAD_POOL, HEAD_HIDDEN),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(HEAD_HIDDEN, num_classes),
    )


class ThinkOrStopCNN(nn.Module):
    """Shared three-stage backbone with one classifier per enabled exit.

    Args:
        in_channels: 1 for MNIST, 3 for CIFAR-10.
        num_classes: number of output logits per exit.
        base_channels: width of stage 1 (stages 2 and 3 use 2x and 4x).
        dropout: dropout probability inside each head.
        multi_exit: if False, only the final (stage-3) head exists.

    ``forward`` returns a list of logits, one per exit, ordered shallow to deep.
    ``exit_stages`` maps list position to the stage index (0-based) of the exit.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 10,
        base_channels: int = 64,
        dropout: float = 0.3,
        multi_exit: bool = True,
    ):
        super().__init__()
        widths = [base_channels * (2**i) for i in range(NUM_STAGES)]
        chans = [in_channels] + widths
        self.stages = nn.ModuleList(make_stage(chans[i], chans[i + 1]) for i in range(NUM_STAGES))
        self.exit_stages: tuple[int, ...] = tuple(range(NUM_STAGES)) if multi_exit else (NUM_STAGES - 1,)
        self.heads = nn.ModuleList(make_head(widths[s], num_classes, dropout) for s in self.exit_stages)
        self.multi_exit = multi_exit
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # Linear layers keep PyTorch's default init, which gives initial
            # logits near zero (initial loss ~ ln(num_classes)).

    @property
    def num_exits(self) -> int:
        return len(self.heads)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run every stage and return logits from every enabled exit."""
        outputs: list[torch.Tensor] = []
        head_iter = iter(zip(self.exit_stages, self.heads))
        next_exit = next(head_iter, None)
        for s, stage in enumerate(self.stages):
            x = stage(x)
            if next_exit is not None and next_exit[0] == s:
                outputs.append(next_exit[1](x))
                next_exit = next(head_iter, None)
        return outputs

    def forward_to_exit(self, x: torch.Tensor, exit_idx: int) -> torch.Tensor:
        """Fixed-exit inference: run only the stages needed for exit ``exit_idx``
        (0-based position in ``self.heads``) and only that exit's head."""
        stage_idx = self.exit_stages[exit_idx]
        for s in range(stage_idx + 1):
            x = self.stages[s](x)
        return self.heads[exit_idx](x)


def build_model(
    dataset_in_channels: int,
    model_type: str,
    num_classes: int = 10,
    base_channels: int = 64,
    dropout: float = 0.3,
) -> ThinkOrStopCNN:
    """Construct the baseline (``"baseline"``) or multi-exit (``"multiexit"``) model."""
    if model_type not in ("baseline", "multiexit"):
        raise ValueError(f"Unknown model type {model_type!r}")
    return ThinkOrStopCNN(
        in_channels=dataset_in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout=dropout,
        multi_exit=model_type == "multiexit",
    )


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# --------------------------------------------------------------------------
# Computation estimate
# --------------------------------------------------------------------------


@dataclass
class ExitCosts:
    """Multiply-accumulate (MAC) counts for one input image.

    Methodology: forward hooks record the output shape of every ``Conv2d`` and
    ``Linear`` layer for a single image; MACs are

    * Conv2d: ``C_out * H_out * W_out * (C_in / groups) * kH * kW``
    * Linear: ``in_features * out_features``

    Batch norm, ReLU, pooling, softmax and the confidence test are ignored
    (they are element-wise and small relative to convolutions). FLOPs are
    reported as ``2 * MACs``. These are *estimates of arithmetic work*, not of
    latency: memory traffic, kernel launches and dynamic batching overheads
    are not modelled.
    """

    stage_macs: list[int]  # per stage
    head_macs: list[int]  # per enabled exit

    def fixed_exit_macs(self, exit_idx: int, exit_stages: tuple[int, ...]) -> int:
        """Cost of fixed-exit inference at ``exit_idx``: stages up to it + that head only."""
        return sum(self.stage_macs[: exit_stages[exit_idx] + 1]) + self.head_macs[exit_idx]

    def adaptive_exit_macs(self, exit_idx: int, exit_stages: tuple[int, ...]) -> int:
        """Cost for a sample that leaves adaptively at ``exit_idx``: all stages up
        to it plus *every* head evaluated on the way (earlier heads that did not
        trigger an exit still cost compute)."""
        return sum(self.stage_macs[: exit_stages[exit_idx] + 1]) + sum(self.head_macs[: exit_idx + 1])

    @property
    def full_macs(self) -> int:
        """Cost of a forward pass through all stages and the final head only."""
        return sum(self.stage_macs) + self.head_macs[-1]


def compute_exit_costs(model: ThinkOrStopCNN, input_shape: tuple[int, int, int]) -> ExitCosts:
    """Count MACs per stage and per head for a single ``(C, H, W)`` input."""
    counts: dict[nn.Module, int] = {}

    def hook(mod: nn.Module, inp, out) -> None:
        if isinstance(mod, nn.Conv2d):
            _, c_out, h, w = out.shape
            k = mod.kernel_size[0] * mod.kernel_size[1]
            counts[mod] = c_out * h * w * (mod.in_channels // mod.groups) * k
        elif isinstance(mod, nn.Linear):
            counts[mod] = mod.in_features * mod.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        with torch.inference_mode():
            model(torch.zeros(1, *input_shape, device=device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    def total(module: nn.Module) -> int:
        return sum(counts.get(m, 0) for m in module.modules())

    return ExitCosts(stage_macs=[total(s) for s in model.stages], head_macs=[total(h) for h in model.heads])
