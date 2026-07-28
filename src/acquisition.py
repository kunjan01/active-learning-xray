"""
acquisition.py
==============
Acquisition functions for active learning: Entropy Sampling and BALD.

The key distinction:
- **Entropy**: "How confused is the model about this image?"
  Uses only the MEAN prediction across MC dropout passes.
  Formula: H[y|x, D] = -sum(p_mean * log(p_mean))

- **BALD** (Bayesian Active Learning by Disagreement):
  "How much would labeling this image reduce our uncertainty about the
  MODEL'S PARAMETERS?" This is mutual information between the prediction
  and the model parameters, estimated via MC dropout.
  Formula: I[y; θ | x, D] = H[y|x, D] - E_θ[H[y|x, θ]]
         = entropy_of_mean - mean_of_entropies

  BALD prefers images where individual dropout masks disagree with each other,
  even if the average prediction looks confident. This is strictly more
  informative than entropy alone for Bayesian model selection.
"""

from typing import List, Tuple

import torch
import numpy as np


# Small constant to avoid log(0)
_EPS = 1e-10


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy along the last dimension.

    Parameters
    ----------
    probs : torch.Tensor
        Probability vectors, shape (..., num_classes).

    Returns
    -------
    torch.Tensor
        Entropy values, shape (...).
    """
    return -(probs * torch.log(probs + _EPS)).sum(dim=-1)


def entropy_score(mc_predictions: torch.Tensor) -> torch.Tensor:
    """
    Entropy-based acquisition score.

    Parameters
    ----------
    mc_predictions : torch.Tensor
        Shape (N, n_passes, num_classes) from MC dropout.

    Returns
    -------
    torch.Tensor
        Entropy scores, shape (N,). Higher = more uncertain.
    """
    # Mean prediction across passes: (N, num_classes)
    mean_probs = mc_predictions.mean(dim=1)
    # Entropy of the mean prediction
    return _entropy(mean_probs)


def bald_score(mc_predictions: torch.Tensor) -> torch.Tensor:
    """
    BALD (Bayesian Active Learning by Disagreement) acquisition score.

    BALD = H[y|x, D] - E_θ[H[y|x, θ]]
         = entropy_of_mean_prediction - mean_of_individual_entropies

    This captures epistemic uncertainty (model parameter uncertainty),
    not just aleatoric uncertainty (inherent label noise). An image gets
    a high BALD score when individual MC dropout passes disagree, even if
    the average prediction looks confident.

    Parameters
    ----------
    mc_predictions : torch.Tensor
        Shape (N, n_passes, num_classes) from MC dropout.
        This is the FULL set of 30 stochastic predictions, not just the mean.

    Returns
    -------
    torch.Tensor
        BALD (mutual information) scores, shape (N,). Higher = more informative.
    """
    # Term 1: Entropy of the mean prediction  H[y|x, D]
    mean_probs = mc_predictions.mean(dim=1)  # (N, num_classes)
    entropy_of_mean = _entropy(mean_probs)  # (N,)

    # Term 2: Mean of entropies of individual passes  E_θ[H[y|x, θ]]
    # _entropy over each pass: (N, n_passes)
    individual_entropies = _entropy(mc_predictions)  # (N, n_passes)
    mean_of_entropies = individual_entropies.mean(dim=1)  # (N,)

    # BALD = mutual information
    bald = entropy_of_mean - mean_of_entropies  # (N,)

    # Clamp to handle numerical noise (MI should be >= 0)
    bald = torch.clamp(bald, min=0.0)

    return bald


def select_top_k(
    scores: torch.Tensor,
    pool_indices: List[int],
    k: int = 25,
) -> List[int]:
    """
    Select the top-k unlabeled images by acquisition score.

    Parameters
    ----------
    scores : torch.Tensor
        Acquisition scores, shape (N,).
    pool_indices : list of int
        The pool indices corresponding to each score.
    k : int
        Number of images to select.

    Returns
    -------
    list of int
        The pool indices of the top-k scoring images.
    """
    k = min(k, len(pool_indices))

    # Get indices of top-k scores
    _, top_k_local = torch.topk(scores, k)
    top_k_local = top_k_local.tolist()

    # Map back to pool indices
    selected = [pool_indices[i] for i in top_k_local]
    return selected


def select_top_k_balanced(
    scores: torch.Tensor,
    pool_indices: List[int],
    pool_labels: List[int],
    k: int = 25,
) -> List[int]:
    """
    Select the top-k unlabeled images, balanced across classes.

    Instead of simply taking the k highest-scoring images (which tends to
    over-select the majority class), this selects the top k/2 from each class.
    This prevents BALD from creating a class-imbalanced labeled set, which
    would cause the model to become biased toward the majority class.

    If one class has fewer than k/2 candidates, the remainder is filled
    from the other class.

    Parameters
    ----------
    scores : torch.Tensor
        Acquisition scores, shape (N,).
    pool_indices : list of int
        The pool indices corresponding to each score.
    pool_labels : list of int
        The true labels for each image in pool_indices (0=NORMAL, 1=PNEUMONIA).
    k : int
        Total number of images to select.

    Returns
    -------
    list of int
        The pool indices of the selected images (balanced across classes).
    """
    k = min(k, len(pool_indices))
    half_k = k // 2

    # Separate indices by class
    class_0_mask = torch.tensor([l == 0 for l in pool_labels], dtype=torch.bool)
    class_1_mask = torch.tensor([l == 1 for l in pool_labels], dtype=torch.bool)

    selected = []

    for mask, n_select in [(class_0_mask, half_k), (class_1_mask, k - half_k)]:
        if mask.sum() == 0:
            continue
        # Get scores for this class only
        class_scores = scores.clone()
        class_scores[~mask] = -float('inf')  # mask out other class

        n_available = mask.sum().item()
        n_to_select = min(n_select, n_available)

        _, top_local = torch.topk(class_scores, n_to_select)
        selected.extend([pool_indices[i] for i in top_local.tolist()])

    # If one class couldn't fill its quota, fill from the other class
    if len(selected) < k:
        remaining_scores = scores.clone()
        selected_set = set(selected)
        # Mask out already-selected indices
        for i, idx in enumerate(pool_indices):
            if idx in selected_set:
                remaining_scores[i] = -float('inf')
        n_remaining = k - len(selected)
        _, top_local = torch.topk(remaining_scores, n_remaining)
        selected.extend([pool_indices[i] for i in top_local.tolist()])

    return selected


def get_acquisition_fn(name: str):
    """
    Factory function to get an acquisition function by name.

    Parameters
    ----------
    name : str
        One of 'bald' or 'entropy'.

    Returns
    -------
    callable
        The acquisition scoring function.
    """
    fns = {
        "bald": bald_score,
        "entropy": entropy_score,
    }
    if name.lower() not in fns:
        raise ValueError(f"Unknown acquisition function: {name}. "
                         f"Choose from: {list(fns.keys())}")
    return fns[name.lower()]
