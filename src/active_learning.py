"""
active_learning.py
==================
The main active learning loop: train → evaluate → score (MC dropout) → select → label → repeat.

This implements the exact pseudocode from the project specification:

    labeled = seed_set (100 images)
    unlabeled = remaining pool
    history = []

    while labeling_budget_remaining:
        train(model, labeled)                      # warm-start
        acc = evaluate(model, fixed_test_set)
        history.append((len(labeled), acc))
        scores = mc_dropout_and_score(model, unlabeled, acquisition_fn=BALD)
        top_k = select_top_k(unlabeled, scores, k=25)
        labeled += reveal_labels(top_k)            # oracle step
        unlabeled -= top_k

    save history

Key design:
- Warm-start: model weights persist across rounds (no re-initialization).
- The test set is NEVER used for selection — only for evaluation after each round.
"""

import json
import os
import time
from typing import Dict, List, Optional

import torch
import numpy as np

from src.dataset import ActiveLearningPool, get_test_dataset
from src.model import create_model, train_one_round
from src.mc_dropout import mc_dropout_predict
from src.acquisition import get_acquisition_fn, select_top_k_balanced
from src.evaluate import evaluate_model, print_evaluation


def run_active_learning(
    data_root: str,
    seed: int,
    acquisition_fn_name: str = "bald",
    budget: int = 500,
    k: int = 25,
    epochs_per_round: int = 7,
    lr: float = 1e-4,
    mc_passes: int = 30,
    save_dir: Optional[str] = None,
    device: torch.device = None,
) -> List[Dict]:
    """
    Run the full active learning loop.

    Parameters
    ----------
    data_root : str
        Path to chest_xray/ root.
    seed : int
        Random seed for reproducibility.
    acquisition_fn_name : str
        Acquisition function: 'bald' or 'entropy'.
    budget : int
        Total labeling budget (including the initial seed set).
    k : int
        Number of images to acquire per round.
    epochs_per_round : int
        Training epochs per AL round.
    lr : float
        Learning rate for Adam.
    mc_passes : int
        Number of MC dropout forward passes.
    save_dir : str or None
        Directory to save results JSON. If None, results are not saved.
    device : torch.device
        Computation device.

    Returns
    -------
    list of dict
        History: one entry per round with n_labeled, accuracy, metrics, losses.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"ACTIVE LEARNING -- seed={seed}, acquisition={acquisition_fn_name}, "
          f"budget={budget}, k={k}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize components
    pool = ActiveLearningPool(data_root, seed=seed, seed_size=100)
    test_dataset = get_test_dataset(data_root)
    use_pin_memory = torch.cuda.is_available()
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=use_pin_memory
    )
    model = create_model(num_classes=2, dropout_p=0.3)
    acquisition_fn = get_acquisition_fn(acquisition_fn_name)

    history: List[Dict] = []
    round_num = 0

    while len(pool.labeled_indices) < budget:
        round_num += 1
        t_start = time.time()

        print(f"\n--- Round {round_num} | Labeled: {len(pool.labeled_indices)} ---")

        # 1. Train on current labeled set (warm-start)
        train_loader = pool.get_labeled_loader(batch_size=32)
        epoch_losses = train_one_round(
            model, train_loader, epochs=epochs_per_round, lr=lr, device=device
        )

        # 2. Evaluate on fixed test set
        metrics = evaluate_model(model, test_loader, device=device)
        print_evaluation(metrics, len(pool.labeled_indices))

        # 3. Record history
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
            "round_time_sec": 0,  # will be updated below
        })

        # 4. Check if we've reached the budget or pool is empty
        if len(pool.labeled_indices) >= budget or len(pool.unlabeled_indices) == 0:
            t_elapsed = time.time() - t_start
            history[-1]["round_time_sec"] = t_elapsed
            print(f"  Budget reached or pool exhausted. Stopping.")
            break

        # 5. MC Dropout scoring on unlabeled pool
        unlabeled_loader = pool.get_unlabeled_loader(batch_size=64)
        mc_predictions, pool_indices = mc_dropout_predict(
            model, unlabeled_loader, n_passes=mc_passes, device=device
        )

        # 6. Compute acquisition scores
        scores = acquisition_fn(mc_predictions)

        # 7. Select top-k (class-balanced to prevent majority-class bias)
        pool_labels = [pool.all_labels[i] for i in pool_indices]
        selected = select_top_k_balanced(scores, pool_indices, pool_labels, k=k)
        print(f"  Selected {len(selected)} images for labeling")

        # 8. Oracle step: reveal labels
        pool.reveal_labels(selected)

        t_elapsed = time.time() - t_start
        history[-1]["round_time_sec"] = t_elapsed
        print(f"  Round time: {t_elapsed:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"active_{acquisition_fn_name}_seed_{seed}.json"
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w") as f:
            json.dump({
                "strategy": f"active_{acquisition_fn_name}",
                "seed": seed,
                "budget": budget,
                "k": k,
                "epochs_per_round": epochs_per_round,
                "mc_passes": mc_passes,
                "history": history,
            }, f, indent=2)
        print(f"\nResults saved to {filepath}")

    print(f"\nFinal accuracy: {history[-1]['accuracy']:.4f} "
          f"with {history[-1]['n_labeled']} labels")

    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run active learning")
    parser.add_argument("--data_root", type=str, default="data/chest_xray",
                        help="Path to chest_xray dataset root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acquisition", type=str, default="bald",
                        choices=["bald", "entropy"])
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--save_dir", type=str, default="experiments/results")
    args = parser.parse_args()

    run_active_learning(
        data_root=args.data_root,
        seed=args.seed,
        acquisition_fn_name=args.acquisition,
        budget=args.budget,
        k=args.k,
        epochs_per_round=args.epochs,
        save_dir=args.save_dir,
    )
