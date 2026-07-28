"""
mc_dropout.py
=============
Monte Carlo Dropout inference for Bayesian uncertainty estimation.

Critical implementation detail:
- model.train() is called (NOT model.eval()) so that dropout remains active.
- Each unlabeled image gets N stochastic forward passes, producing N different
  softmax probability vectors due to different dropout masks.
- The FULL set of N predictions is returned (not just the mean), because BALD
  needs the individual-pass entropies to compute mutual information.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def mc_dropout_predict(
    model: nn.Module,
    dataloader: DataLoader,
    n_passes: int = 30,
    device: torch.device = None,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Run N stochastic forward passes over the unlabeled pool.

    Parameters
    ----------
    model : nn.Module
        Trained model with dropout layer(s).
    dataloader : DataLoader
        DataLoader over unlabeled images. Must return (images, pool_indices).
    n_passes : int
        Number of stochastic forward passes (default: 30).
    device : torch.device
        Computation device (auto-detected if None).

    Returns
    -------
    mc_predictions : torch.Tensor
        Shape (N_images, n_passes, num_classes). The full set of softmax
        probability vectors for each image across all passes.
    pool_indices : list of int
        The corresponding pool indices, in the same order as mc_predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # CRITICAL: Keep model in train mode so dropout stays active.
    # This is the entire point of MC dropout — without this, all N passes
    # would produce identical outputs and uncertainty estimates would be zero.
    model.train()

    all_predictions = []  # Will hold per-batch results
    all_indices = []

    with torch.no_grad():  # No gradients needed — this is inference
        for images, indices in tqdm(dataloader, desc="MC Dropout", leave=False):
            images = images.to(device)
            batch_size = images.size(0)

            # Collect N stochastic forward passes for this batch
            batch_preds = []
            for _ in range(n_passes):
                logits = model(images)  # (batch_size, num_classes)
                probs = F.softmax(logits, dim=-1)  # (batch_size, num_classes)
                batch_preds.append(probs.cpu())

            # Stack: (n_passes, batch_size, num_classes) -> (batch_size, n_passes, num_classes)
            batch_preds = torch.stack(batch_preds, dim=0).permute(1, 0, 2)
            all_predictions.append(batch_preds)

            # Collect pool indices
            if isinstance(indices, torch.Tensor):
                all_indices.extend(indices.tolist())
            else:
                all_indices.extend(indices)

    # Concatenate all batches: (total_images, n_passes, num_classes)
    mc_predictions = torch.cat(all_predictions, dim=0)

    return mc_predictions, all_indices
