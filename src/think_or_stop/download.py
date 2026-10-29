"""Download datasets explicitly (the only module that makes network calls).

    python -m think_or_stop.download --dataset mnist cifar10
    python -m think_or_stop.download --dataset cifar10c   # optional, ~2.9 GB

MNIST and CIFAR-10 come from the official torchvision download mechanism
(which verifies MD5 checksums). CIFAR-10-C comes from its authors' Zenodo
record and is checked against the MD5 published in that record.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from .datasets import cifar10c_metadata, download_cifar10c, load_official


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m think_or_stop.download", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", choices=("mnist", "cifar10", "cifar10c"), default=["mnist", "cifar10"])
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--info-only", action="store_true", help="for cifar10c: print source URL/checksum, do not download")
    args = p.parse_args(argv)

    for name in args.dataset:
        if name == "cifar10c":
            meta = cifar10c_metadata()
            print("CIFAR-10-C source:", meta)
            if not args.info_only:
                path = download_cifar10c(args.data_dir)
                print(f"cifar10c: ready in {path}")
            continue
        train = load_official(name, args.data_dir, True, None, download=True)
        test = load_official(name, args.data_dir, False, None, download=True)
        print(f"{name}: {len(train)} train / {len(test)} test images in {args.data_dir}")


if __name__ == "__main__":
    main()
