"""
model.py
========
ResNet18 setup with dropout layer for MC dropout inference.

Design decisions:
- Freeze everything except layer4 + FC (fine-tuning, not full training).
- nn.Dropout(p=0.3) inserted BEFORE the final FC — this is what enables
  MC dropout at inference time. Without this specific placement, the
  stochastic forward passes in mc_dropout.py would be deterministic.
- Warm-start: train_one_round() updates the existing model weights rather
  than reinitializing, so the model improves incrementally across AL rounds.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


def create_model(num_classes: int = 2, dropout_p: float = 0.3) -> nn.Module:
    """
    Create a ResNet18 pretrained on ImageNet with:
    1. All layers frozen except layer4 and the new FC head.
    2. A Dropout layer before the final linear classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes (2 for NORMAL vs PNEUMONIA).
    dropout_p : float
        Dropout probability for MC dropout.

    Returns
    -------
    nn.Module
        The configured ResNet18 model.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final FC layer with Dropout + Linear
    in_features = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes),
    )
    # The new FC layer's parameters have requires_grad=True by default

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Model] ResNet18: {n_trainable:,} / {n_total:,} parameters trainable "
          f"({100 * n_trainable / n_total:.1f}%)")

    return model


def train_one_round(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 7,
    lr: float = 1e-4,
    device: torch.device = None,
) -> List[float]:
    """
    Train the model for one active learning round (warm-start).

    Parameters
    ----------
    model : nn.Module
        The model to train (modified in-place).
    train_loader : DataLoader
        DataLoader over the current labeled set.
    epochs : int
        Number of training epochs for this round.
    lr : float
        Learning rate for Adam optimizer.
    device : torch.device
        Device to train on (auto-detected if None).

    Returns
    -------
    list of float
        Average training loss for each epoch (for the report appendix).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    epoch_losses: List[float] = []

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

    return epoch_losses
