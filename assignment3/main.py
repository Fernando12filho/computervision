"""CAP 5410 - Assignment 3: run every experiment end to end.

    python main.py              train all configurations, pick one on validation, test it once, make figures
    python main.py --skip-train reuse the saved results/ and only redo selection, test and figures

Steps:
  1. Print the environment and the per-layer parameter table.
  2. Train the baseline and the learning-rate alternatives for 10 epochs each.
  3. Select the configuration with the best final validation accuracy
     (the test set is not touched until this choice is made).
  4. Evaluate the selected model once on the 10,000 test images.
  5. Save every figure into report/figures/.
"""
import argparse
import json
import os
import platform
import sys

import matplotlib
import numpy as np
import sklearn
import torch
import torchvision

import plots
from model import Net, count_parameters, layer_table
from train import RESULTS_DIR, run_experiment, test_model

SEED = 42
BASE = {"optimizer": "SGD", "momentum": 0.9, "batch_size": 64, "epochs": 10, "seed": SEED}

# Task 4 factor: learning rate. Everything else stays the same as the baseline.
CONFIGS = [
    {**BASE, "name": "lr_0.01", "lr": 0.01, "baseline": True},
    {**BASE, "name": "lr_0.001", "lr": 0.001},
    {**BASE, "name": "lr_0.1", "lr": 0.1},
]
BASELINE = "lr_0.01"


class Logger:
    """Print to the console and keep a copy in results/output_log.txt."""

    def __init__(self, path, mode):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, mode, encoding="utf-8")

    def __call__(self, msg=""):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()


def environment(log):
    log("=== Environment")
    log(f"Python       {sys.version.split()[0]}")
    log(f"PyTorch      {torch.__version__}")
    log(f"torchvision  {torchvision.__version__}")
    log(f"NumPy        {np.__version__}")
    log(f"scikit-learn {sklearn.__version__}")
    log(f"Matplotlib   {matplotlib.__version__}")
    log(f"OS           {platform.platform()}")
    log(f"Processor    {platform.processor()}")
    log(f"Device       CPU (CUDA available: {torch.cuda.is_available()}), {torch.get_num_threads()} threads")
    log(f"Seed         {SEED}")


def parameter_table(log):
    log("\n=== Layers (input -> output shape, trainable parameters)")
    model = Net()
    for name, inp, out, n in layer_table(model):
        log(f"{name:6s} {str(inp):15s} -> {str(out):15s} {n:7d}")
    log(f"total trainable parameters: {count_parameters(model)}")


def select(log):
    """Pick the configuration with the highest validation accuracy after epoch 10 (validation data only)."""
    log("\n=== Validation summary after epoch 10")
    best, best_key = None, None
    for cfg in CONFIGS:
        with open(os.path.join(RESULTS_DIR, cfg["name"], "history.json")) as f:
            e = json.load(f)["epoch"]
        key = (e["val_acc"][-1], -e["val_loss"][-1])     # tie-break on lower validation loss
        log(f"{cfg['name']:9s} val acc {e['val_acc'][-1]:.4f}  val F1 {e['val_f1'][-1]:.4f}  "
            f"val loss {e['val_loss'][-1]:.4f}  best val acc {max(e['val_acc']):.4f} "
            f"(epoch {int(np.argmax(e['val_acc'])) + 1})")
        if best is None or key > best_key:
            best, best_key = cfg, key
    log(f"selected configuration: {best['name']}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="reuse saved histories and weights")
    args = parser.parse_args()

    log = Logger(os.path.join(RESULTS_DIR, "output_log.txt"), "a" if args.skip_train else "w")
    environment(log)
    parameter_table(log)

    if not args.skip_train:
        for cfg in CONFIGS:
            run_experiment(cfg, log)

    chosen = select(log)
    test_model(chosen, log)
    plots.make_all([c["name"] for c in CONFIGS], BASELINE)
    log(f"\nfigures saved to {plots.FIG_DIR}")


if __name__ == "__main__":
    main()
