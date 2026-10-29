# Think-or-Stop

Think-or-Stop is an experimental PyTorch project for studying adaptive
early-exit image classification. It compares a standard CNN with a
three-stage multi-exit CNN that can stop inference early when an intermediate
prediction is sufficiently confident.

The project measures the accuracy, calibration, robustness, computation,
exit-depth distribution, and latency trade-offs of fixed-exit and adaptive
inference.

## Features

- MNIST and CIFAR-10 training and evaluation
- Standard final-exit CNN baseline
- Three-exit CNN with configurable loss weights
- Confidence-threshold adaptive inference
- Validation-only threshold selection
- Synthetic Gaussian-noise and blur corruptions
- Optional CIFAR-10-C robustness evaluation
- Accuracy, ECE, estimated MACs, exit-depth, and latency metrics
- PNG plots and multi-seed Markdown reports
- CPU and CUDA execution

## Requirements

- Python 3.10 or newer
- PyTorch 2.2 or newer
- torchvision 0.17 or newer
- NumPy, Matplotlib, and optionally pytest
- A CUDA-capable GPU is recommended for the full CIFAR-10 experiment

The tested environment is recorded in [`requirements.txt`](requirements.txt).
When using CUDA, install the matching PyTorch and torchvision wheels from the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/)
before installing the remaining requirements. For example:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install -e .
```

For a CPU-only environment, install the CPU-compatible PyTorch wheels for your
platform instead.

## Quick start

Create and activate a virtual environment from the project root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Download a dataset:

```powershell
python -m think_or_stop.download --dataset mnist
python -m think_or_stop.download --dataset cifar10
```

Run a small CPU smoke test:

```powershell
python -m think_or_stop.train `
  --dataset mnist `
  --model multiexit `
  --device cpu `
  --epochs 1 `
  --max-train-batches 10 `
  --num-workers 0
```

The default training run uses the `cuda` device. Pass `--device cpu` when no
CUDA device is available.

## Full CIFAR-10 experiment

[`run_all.ps1`](run_all.ps1) runs the complete pipeline:

1. Downloads CIFAR-10.
2. Verifies the GPU and runs a forward/backward smoke test.
3. Trains baseline and multi-exit models for each seed.
4. Evaluates clean accuracy, synthetic corruptions, adaptive thresholds, and
   latency.
5. Generates figures.
6. Aggregates results across seeds.

Run the default three-seed, 30-epoch experiment:

```powershell
.\run_all.ps1
```

Run one seed or reduce the epoch count while iterating:

```powershell
.\run_all.ps1 -Seeds 0 -Epochs 1
```

If PowerShell execution policy prevents the script from running:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

Useful parameters include `-Seeds`, `-Epochs`, `-BatchSize`, `-Threshold`,
and `-MaxDrop`.

## Individual commands

Train a model:

```powershell
python -m think_or_stop.train `
  --dataset cifar10 `
  --model multiexit `
  --device cuda `
  --epochs 30 `
  --batch-size 128 `
  --seed 0
```

The supported model names are `baseline` and `multiexit`. Training writes
`best.pt` and `last.pt` under `checkpoints/<run>/`, plus logs and configuration
metadata under `results/logs/<run>/`.

Evaluate a multi-exit checkpoint:

```powershell
python -m think_or_stop.evaluate `
  --checkpoint checkpoints/cifar10_multiexit_s0/best.pt `
  --device cuda `
  --threshold 0.9 `
  --select-threshold `
  --constraint relative `
  --max-drop 0.01 `
  --corruptions synthetic
```

Thresholds are selected using the validation split and then applied unchanged
to the test and corruption datasets. Use `--corruptions cifar10c` to evaluate
the optional CIFAR-10-C dataset. Its source and checksum information can be
displayed without downloading:

```powershell
python -m think_or_stop.download --dataset cifar10c --info-only
```

Generate plots from evaluation metrics:

```powershell
python -m think_or_stop.plots `
  --metrics-dir results/metrics/cifar10_multiexit_s0 `
  --baseline-dir results/metrics/cifar10_baseline_s0
```

Aggregate multiple runs:

```powershell
python -m think_or_stop.report `
  --metrics-dirs `
    results/metrics/cifar10_baseline_s0 `
    results/metrics/cifar10_multiexit_s0 `
  --out results/metrics/report_cifar10
```

Run the test suite:

```powershell
python -m pytest
```

Some tests require a downloaded dataset or CUDA and are marked accordingly.

## Project layout

```text
src/think_or_stop/       Package source and command-line modules
tests/                   Unit and integration tests
notebooks/               Training walkthrough
data/raw/                Downloaded datasets
checkpoints/             Saved model checkpoints
results/logs/            Training and evaluation logs
results/metrics/         CSV, JSON, and Markdown metrics
results/figures/         Generated plots
docs/                    Project report and presentation
run_all.ps1              End-to-end CIFAR-10 pipeline
```

Downloaded datasets, checkpoints, logs, metrics, and figures are generated
artifacts and are ignored by Git. See the
[training walkthrough](notebooks/ThinkOrStop_Training_Walkthrough.ipynb) for a
step-by-step exploration of the model and pipeline.

## Current reference results

The checked-in aggregate CIFAR-10 report is available at
[`results/metrics/report_cifar10.md`](results/metrics/report_cifar10.md).
Across three seeds, the reference multi-exit run reports 89.28% test accuracy,
63.5% of full-depth estimated MACs, and a mean exit depth of 2.01 for the
validation-selected adaptive policy. These values are descriptive results for
the recorded environment, not a statistical significance claim.

## License and attribution

The repository does not currently declare a software license. CIFAR-10-C, when
used, is attributed to Hendrycks and Dietterich's *Benchmarking Neural Network
Robustness to Common Corruptions and Perturbations* and is distributed under
the terms provided by its authors.
