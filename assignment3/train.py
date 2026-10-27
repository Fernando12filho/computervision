"""CAP 5410 - Assignment 3: training and evaluation loops.

`run_experiment` trains one configuration for a fixed number of epochs and
returns everything the report needs: batch-level metrics for the first
epoch and epoch-level metrics for the training and validation sets.
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from dataloader import get_loaders
from model import Net

RESULTS_DIR = "./results"


def set_seed(seed):
    """Seed every random number generator used during training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def macro_f1(y_true, y_pred):
    # Unweighted mean of the per-class F1-scores over the digits that appear in y_true or y_pred.
    # On the full training/validation/test sets this is all 10 digits; a 64-image batch can miss
    # a digit, and scoring that absent digit as F1 = 0 would wrongly pull the batch's score down.
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def evaluate(model, loader, criterion):
    """Loss, accuracy, macro-F1 and the raw predictions over a whole loader (eval mode, no gradients)."""
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            y_true.append(labels)
            y_pred.append(outputs.argmax(1))
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return {
        "loss": total_loss / len(y_true),
        "acc": float((y_true == y_pred).mean()),
        "f1": float(macro_f1(y_true, y_pred)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def make_optimizer(name, params, lr, momentum):
    if name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"unknown optimizer {name}")


def run_experiment(cfg, log=print):
    """Train `cfg` and save its history and final weights under results/<name>."""
    set_seed(cfg["seed"])
    trainloader, valloader, _, (mean, std) = get_loaders(cfg["batch_size"], cfg["seed"])
    # unshuffled copy of the training split, used only to measure epoch-level training metrics
    train_eval_loader = DataLoader(trainloader.dataset, batch_size=1000, shuffle=False)

    model = Net()           # created right after seeding: same initial weights for every config
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["momentum"])

    log(f"\n=== {cfg['name']}: {cfg['optimizer']} lr={cfg['lr']} momentum={cfg['momentum']} "
        f"batch={cfg['batch_size']} epochs={cfg['epochs']} seed={cfg['seed']}")
    log(f"normalization (training split): mean={mean:.4f} std={std:.4f}")

    batch_hist = {"batch": [], "loss": [], "acc": [], "f1": [], "f1_cum": []}
    epoch_hist = {k: [] for k in ["epoch", "train_loss", "train_acc", "train_f1",
                                  "val_loss", "val_acc", "val_f1", "time"]}

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        start = time.time()
        seen_true, seen_pred = [], []
        for i, (images, labels) in enumerate(trainloader, start=1):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if epoch == 1:
                # batch-level record for Task 2: metrics of this batch's own predictions
                preds = outputs.argmax(1)
                seen_true.append(labels)
                seen_pred.append(preds)
                batch_hist["batch"].append(i)
                batch_hist["loss"].append(loss.item())
                batch_hist["acc"].append((preds == labels).float().mean().item())
                batch_hist["f1"].append(macro_f1(labels.numpy(), preds.numpy()))
                # cumulative macro-F1 over every training image seen so far in epoch 1
                batch_hist["f1_cum"].append(macro_f1(torch.cat(seen_true).numpy(),
                                                     torch.cat(seen_pred).numpy()))

        train_time = time.time() - start
        # epoch-level summaries: both sets are re-evaluated in eval mode with the end-of-epoch weights
        tr = evaluate(model, trainloader, criterion)
        va = evaluate(model, valloader, criterion)
        for key, value in [("epoch", epoch), ("train_loss", tr["loss"]), ("train_acc", tr["acc"]),
                           ("train_f1", tr["f1"]), ("val_loss", va["loss"]), ("val_acc", va["acc"]),
                           ("val_f1", va["f1"]), ("time", train_time)]:
            epoch_hist[key].append(value)
        log(f"epoch {epoch:2d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} F1 {tr['f1']:.4f} "
            f"| val loss {va['loss']:.4f} acc {va['acc']:.4f} F1 {va['f1']:.4f} | {train_time:.1f}s")

    out_dir = os.path.join(RESULTS_DIR, cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({"config": cfg, "norm": [mean, std], "batch": batch_hist, "epoch": epoch_hist}, f, indent=1)
    return model


def test_model(cfg, log=print):
    """Evaluate the saved weights of `cfg` once on the 10,000 MNIST test images."""
    _, _, testloader, _ = get_loaders(cfg["batch_size"], cfg["seed"])
    model = Net()
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, cfg["name"], "model.pt"), weights_only=True))
    res = evaluate(model, testloader, nn.CrossEntropyLoss())
    y_true, y_pred = res["y_true"], res["y_pred"]
    metrics = {
        "config": cfg["name"],
        "test_loss": res["loss"],
        "test_acc": res["acc"],
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": res["f1"],
        "per_class_f1": f1_score(y_true, y_pred, average=None).tolist(),
    }
    log(f"\n=== TEST ({cfg['name']}, evaluated once)")
    for k in ["test_loss", "test_acc", "macro_precision", "macro_recall", "macro_f1"]:
        log(f"{k:16s} {metrics[k]:.4f}")
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    np.save(os.path.join(RESULTS_DIR, "test_predictions.npy"), np.stack([y_true, y_pred]))
    return metrics
