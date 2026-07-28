"""
baseline.py
===========
Random sampling baseline: identical to the active learning loop but replaces
the MC dropout + acquisition step with uniform random selection.

This is NOT optional — the entire project's evidence rests on comparing
active learning (BALD) against this random baseline across multiple seeds.
Without this, we have no evidence that BALD is actually better than chance.
"""

import json
import os
import random
import time
from typing import Dict, List, Optional

import torch
import numpy as np

from src.dataset import ActiveLearningPool, get_test_dataset
from src.model import create_model, train_one_round
from src.evaluate import evaluate_model, print_evaluation


def run_random_baseline(
    data_root: str,
    seed: int,
    budget: int = 500,
    k: int = 25,
    epochs_per_round: int = 7,
    lr: float = 1e-4,
    save_dir: Optional[str] = None,
    device: torch.device = None,
) -> List[Dict]:
    """
    Run the random sampling baseline.

    Identical to active_learning.run_active_learning() except:
    - No MC dropout inference (saves compute).
    - Selection is random.sample(unlabeled_indices, k) instead of top-k by score.

    Parameters
    ----------
    data_root : str
        Path to chest_xray/ root.
    seed : int
        Random seed (same seeds as active learning for paired comparison).
    budget : int
        Total labeling budget.
    k : int
        Number of images to randomly sample per round.
    epochs_per_round : int
        Training epochs per round.
    lr : float
        Learning rate.
    save_dir : str or None
        Directory to save results.
    device : torch.device
        Computation device.

    Returns
    -------
    list of dict
        History with same format as active_learning.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"RANDOM BASELINE -- seed={seed}, budget={budget}, k={k}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    # Initialize components (same as active learning)
    pool = ActiveLearningPool(data_root, seed=seed, seed_size=100)
    test_dataset = get_test_dataset(data_root)
    use_pin_memory = torch.cuda.is_available()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=use_pin_memory
    )
    model = create_model(num_classes=2, dropout_p=0.3)

    history: List[Dict] = []
    round_num = 0

    while len(pool.labeled_indices) < budget:
        round_num += 1
        t_start = time.time()

        print(f"\n--- Round {round_num} | Labeled: {len(pool.labeled_indices)} ---")

        # 1. Train on current labeled set (warm-start, same as AL)
        train_loader = pool.get_labeled_loader(batch_size=32)
        epoch_losses = train_one_round(
            model, train_loader, epochs=epochs_per_round, lr=lr, device=device
        )

        # 2. Evaluate on fixed test set (same as AL)
        metrics = evaluate_model(model, test_loader, device=device)
        print_evaluation(metrics, len(pool.labeled_indices))

        # 3. Record history (same format as AL for easy comparison)
        stats = pool.get_stats()
        history.append({
            "round": round_num,
            "n_labeled": len(pool.labeled_indices),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "epoch_losses": epoch_losses,
            "pool_stats": stats,
            "round_time_sec": 0,
        })

        # 4. Check stopping condition
        if len(pool.labeled_indices) >= budget or len(pool.unlabeled_indices) == 0:
            t_elapsed = time.time() - t_start
            history[-1]["round_time_sec"] = t_elapsed
            print(f"  Budget reached or pool exhausted. Stopping.")
            break

        # 5. RANDOM selection (this is the ONLY difference from active learning)
        n_select = min(k, len(pool.unlabeled_indices))
        selected = rng.sample(pool.unlabeled_indices, n_select)
        print(f"  Randomly selected {len(selected)} images for labeling")

        # 6. Oracle step: reveal labels
        pool.reveal_labels(selected)

        t_elapsed = time.time() - t_start
        history[-1]["round_time_sec"] = t_elapsed
        print(f"  Round time: {t_elapsed:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"random_seed_{seed}.json"
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w") as f:
            json.dump({
                "strategy": "random",
                "seed": seed,
                "budget": budget,
                "k": k,
                "epochs_per_round": epochs_per_round,
                "history": history,
            }, f, indent=2)
        print(f"\nResults saved to {filepath}")

    print(f"\nFinal accuracy: {history[-1]['accuracy']:.4f} "
          f"with {history[-1]['n_labeled']} labels")

    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run random baseline")
    parser.add_argument("--data_root", type=str, default="data/chest_xray")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--save_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    run_random_baseline(
        data_root=args.data_root,
        seed=args.seed,
        budget=args.budget,
        k=args.k,
        epochs_per_round=args.epochs,
        save_dir=args.save_dir,
    )
