"""Classify your own image files with a trained checkpoint.

    python -m think_or_stop.predict photos/cat.jpg photos/ --checkpoint checkpoints/cifar10_multiexit_s0/best.pt

Images are converted to the training format (RGB 32x32 for CIFAR-10,
grayscale 28x28 for MNIST) and normalised with the training statistics.
Multi-exit models use confidence-based early exiting; by default the threshold
selected on validation data by ``evaluate`` is used if it exists, else 0.9.

The model only knows its 10 training classes and was trained on tiny
low-resolution images, so it will force any input into one of those classes
and is much less accurate on arbitrary photos than on the CIFAR-10 test set.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torchvision import transforms as T

from .evaluate import load_model
from .inference import adaptive_predict
from .utils import read_json, resolve_device

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def find_images(paths: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")
    if not files:
        raise FileNotFoundError("No image files found")
    return files


def default_threshold(run_name: str, output_dir: str) -> tuple[float, str]:
    sel = Path(output_dir) / "results" / "metrics" / run_name / "threshold_selection.json"
    if sel.is_file():
        return float(read_json(sel)["selected_threshold"]), f"selected on validation data ({sel})"
    return 0.9, "default 0.9 (no threshold_selection.json found)"


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m think_or_stop.predict", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", help="image files and/or folders of images")
    p.add_argument("--checkpoint", default="checkpoints/cifar10_multiexit_s0/best.pt")
    p.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu")
    p.add_argument("--threshold", type=float, default=None,
                   help="early-exit confidence threshold (default: validation-selected, else 0.9)")
    p.add_argument("--full-depth", action="store_true", help="never exit early (always use the final exit)")
    p.add_argument("--top-k", type=int, default=3, help="also show the k most likely classes")
    p.add_argument("--invert", action="store_true",
                   help="invert colours (MNIST expects a light digit on a dark background)")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    model, cfg, _ = load_model(args.checkpoint, device)
    spec = cfg.spec
    if args.full_depth:
        thr, why = math.inf, "full depth (--full-depth)"
    elif args.threshold is not None:
        thr, why = args.threshold, "given with --threshold"
    else:
        thr, why = default_threshold(cfg.run_name, cfg.output_dir)
    if model.num_exits > 1:
        print(f"Model: {cfg.model} on {cfg.dataset}, threshold {thr:g} [{why}], device {device}")
    else:
        print(f"Model: {cfg.model} on {cfg.dataset} (single exit), device {device}")

    tf = T.Compose([T.Resize((spec.image_size, spec.image_size)), T.ToTensor(), T.Normalize(spec.mean, spec.std)])
    mode = "RGB" if spec.in_channels == 3 else "L"
    exit_names = [f"exit {s + 1}" for s in model.exit_stages]

    for path in find_images(args.images):
        img = Image.open(path).convert(mode)
        if args.invert:
            img = ImageOps.invert(img)
        x = tf(img).unsqueeze(0).to(device)
        r = adaptive_predict(model, x, thr)
        e = int(r.exit_indices[0])
        # Top-k from the exit that made the decision (recomputed for display only).
        with torch.inference_mode():
            probs = F.softmax(model.forward_to_exit(x, e).float(), dim=1)[0]
        top = probs.topk(min(args.top_k, spec.num_classes))
        alts = ", ".join(f"{spec.classes[i]} {v:.1%}" for v, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"{path.name}: {spec.classes[int(r.predictions[0])]} "
              f"(confidence {float(r.confidences[0]):.1%}, decided at {exit_names[e]}) | top-{len(top.values)}: {alts}")


if __name__ == "__main__":
    main()
