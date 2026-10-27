"""CAP 5410 - Assignment 3: figures for the report.

Reads results/<config>/history.json and
writes every figure into report/figures/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "./results"
FIG_DIR = "./report/figures"

# fixed colors per configuration (color follows the config, never its rank)
COLORS = {"lr_0.01": "#2a78d6", "lr_0.001": "#eb6834", "lr_0.1": "#1baf7a"}
INK = "#52514e"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": INK, "axes.labelcolor": "#0b0b0b",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
    "lines.linewidth": 2,
})


def load(name):
    with open(os.path.join(RESULTS_DIR, name, "history.json")) as f:
        return json.load(f)


def label(h):
    c = h["config"]
    return f"{c['optimizer']}, lr={c['lr']}" + (" (baseline)" if c.get("baseline") else "")


def moving_average(x, w=50):
    """Trailing moving average; the first w-1 points average whatever is available."""
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
# Task 2: first epoch, batch level (baseline)
# ---------------------------------------------------------------

def plot_first_epoch(h, step=5, window=50):
    b = h["batch"]
    color = COLORS[h["config"]["name"]]
    batches = np.array(b["batch"])
    bs = h["config"]["batch_size"]
    cfg_txt = f"{label(h)}, batch size {bs}"
    specs = [
        ("loss", "Training loss (cross-entropy)", "epoch1_loss.png"),
        ("acc", "Training accuracy", "epoch1_accuracy.png"),
    ]
    for key, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot(batches[::step], np.array(b[key])[::step], color=color, alpha=0.3, linewidth=1,
                label=f"per batch (every {step}th batch)")
        ax.plot(batches, moving_average(b[key], window), color=color,
                label=f"running average ({window} batches)")
        ax.set_xlabel("Batch index within epoch 1")
        ax.set_ylabel(ylabel)
        if key == "acc":
            ax.set_ylim(0, 1.02)
        ax.set_title(f"Epoch 1, {ylabel.split(' (')[0].lower()} by batch ({cfg_txt})", fontsize=10)
        ax.legend(loc="center right")
        save(fig, fname)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(batches[::step], np.array(b["f1"])[::step], color=color, alpha=0.3, linewidth=1,
            label=f"macro-F1 of the batch (every {step}th batch)")
    ax.plot(batches, moving_average(b["f1"], window), color=color,
            label=f"running average ({window} batches)")
    ax.plot(batches, b["f1_cum"], color=INK, linestyle="--",
            label="cumulative macro-F1 (all images so far)")
    ax.set_xlabel("Batch index within epoch 1")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Epoch 1, macro-F1 by batch ({cfg_txt})", fontsize=10)
    ax.legend(loc="center right")
    save(fig, "epoch1_f1.png")


# ---------------------------------------------------------------
# Task 3: ten epochs, epoch level (baseline)
# ---------------------------------------------------------------

def plot_epochs(h):
    e = h["epoch"]
    color = COLORS[h["config"]["name"]]
    for key, ylabel, fname in [("loss", "Loss (cross-entropy)", "epochs_loss.png"),
                               ("acc", "Accuracy", "epochs_accuracy.png"),
                               ("f1", "Macro-F1", "epochs_f1.png")]:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot(e["epoch"], e[f"train_{key}"], color=color, marker="o", markersize=4, label="training")
        ax.plot(e["epoch"], e[f"val_{key}"], color=color, marker="s", markersize=4,
                linestyle="--", label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(e["epoch"])
        ax.set_title(f"Training vs. validation {ylabel.split(' (')[0].lower()} ({label(h)}, "
                     f"batch size {h['config']['batch_size']})", fontsize=10)
        ax.legend()
        save(fig, fname)


# ---------------------------------------------------------------
# Task 4: learning-rate comparison
# ---------------------------------------------------------------

def plot_comparison(histories):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, ax, ylabel in [("loss", axes[0], "Loss (cross-entropy)"), ("acc", axes[1], "Accuracy")]:
        for h in histories:
            e = h["epoch"]
            color = COLORS[h["config"]["name"]]
            ax.plot(e["epoch"], e[f"train_{key}"], color=color, label=f"{label(h)}, training")
            ax.plot(e["epoch"], e[f"val_{key}"], color=color, linestyle="--",
                    label=f"{label(h)}, validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 11))
        ax.set_title(f"{ylabel.split(' (')[0]} per epoch", fontsize=10)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Loss (cross-entropy, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    save(fig, "compare_curves.png")

    # validation accuracy zoomed in, only for the runs that converged (a diverged run sits near 10%)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for h in histories:
        e = h["epoch"]
        if e["val_acc"][-1] < 0.5:
            continue
        ax.plot(e["epoch"], np.array(e["val_acc"]) * 100, color=COLORS[h["config"]["name"]],
                marker="o", markersize=4, label=label(h))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(96, 100)
    ax.set_title("Validation accuracy per epoch, converged runs only (zoomed to 96-100%)", fontsize=10)
    ax.legend(loc="lower right")
    save(fig, "compare_val_acc.png")

    # first-epoch training loss for every configuration
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for h in histories:
        b = h["batch"]
        ax.plot(b["batch"], moving_average(b["loss"], 50), color=COLORS[h["config"]["name"]], label=label(h))
    ax.set_xlabel("Batch index within epoch 1")
    ax.set_ylabel("Training loss (50-batch running avg.)")
    ax.set_title("Epoch 1 training loss by learning rate", fontsize=10)
    ax.legend()
    save(fig, "compare_epoch1_loss.png")


def make_all(names, baseline):
    histories = [load(n) for n in names]
    base = load(baseline)
    plot_first_epoch(base)
    plot_epochs(base)
    plot_comparison(histories)


if __name__ == "__main__":
    from main import CONFIGS, BASELINE
    make_all([c["name"] for c in CONFIGS], BASELINE)
