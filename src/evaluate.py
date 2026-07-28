"""
evaluate.py
===========
Evaluation utilities for the fixed held-out test set.

This module ONLY evaluates on the test set that was carved out at the start.
It is never used on training or pool data. The model is set to eval() mode
for deterministic inference (no dropout), unlike mc_dropout.py which keeps
dropout active.
"""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate the model on the fixed test set.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    test_loader : DataLoader
        DataLoader over the held-out test set.
    device : torch.device
        Computation device (auto-detected if None).

    Returns
    -------
    dict
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
        All metrics use binary averaging (positive class = PNEUMONIA = 1).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()  # Deterministic inference — dropout OFF

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="binary",
                                     pos_label=1, zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="binary",
                               pos_label=1, zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="binary",
                       pos_label=1, zero_division=0),
    }

    return metrics


def print_evaluation(metrics: Dict[str, float], n_labeled: int) -> None:
    """Pretty-print evaluation results for a single round."""
    print(f"  [Eval] Labels={n_labeled:>4d} | "
          f"Acc={metrics['accuracy']:.4f} | "
          f"Prec={metrics['precision']:.4f} | "
          f"Rec={metrics['recall']:.4f} | "
          f"F1={metrics['f1']:.4f}")
