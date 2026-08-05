"""
experiments/demo_run.py
=======================
Quick demonstration script for VIVA/presentation.
Runs a MINI experiment (1 seed, 2 rounds, fewer MC passes) to show the
pipeline works end-to-end in ~5-10 minutes instead of 15 hours.

Usage:
    python experiments/demo_run.py --data_root data/chest_xray
"""

import argparse
import os
import sys
import time
from datetime import timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.active_learning import run_active_learning
from src.baseline import run_random_baseline


def run_demo(data_root: str):
    """Run a quick demo: 1 seed, budget=150 (2 rounds), mc_passes=5."""
    print("=" * 60)
    print("  QUICK DEMO - Active Learning vs Random")
    print("  Config: 1 seed, budget=150, k=25, mc_passes=5")
    print("  Estimated time: 5-10 minutes on CPU")
    print("=" * 60)

    start = time.time()

    # --- BALD Active Learning ---
    print("\n[1/2] Running BALD Active Learning (seed=42)...")
    print("-" * 60)
    bald_history = run_active_learning(
        data_root=data_root,
        seed=42,
        acquisition_fn_name="bald",
        budget=150,        # 2 rounds (100 -> 125 -> 150)
        k=25,
        epochs_per_round=5,  # Fewer epochs for speed
        mc_passes=5,         # Fewer MC passes for speed
        save_dir=None,       # Don't overwrite saved results
    )

    # --- Random Baseline ---
    print("\n[2/2] Running Random Baseline (seed=42)...")
    print("-" * 60)
    random_history = run_random_baseline(
        data_root=data_root,
        seed=42,
        budget=150,
        k=25,
        epochs_per_round=5,
        save_dir=None,
    )

    # --- Summary ---
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE!")
    print(f"  Total time: {timedelta(seconds=int(elapsed))}")
    print("=" * 60)

    print("\n--- Results Comparison ---")
    print(f"{'Round':<8} {'Labels':<8} {'BALD Acc':<12} {'Random Acc':<12} {'Winner':<10}")
    print("-" * 50)
    for b, r in zip(bald_history, random_history):
        bald_acc = b["accuracy"]
        rand_acc = r["accuracy"]
        winner = "BALD" if bald_acc > rand_acc else "Random"
        print(f"{b['round']:<8} {b['n_labeled']:<8} {bald_acc:<12.4f} {rand_acc:<12.4f} {winner:<10}")

    print("\n--- Class Balance (BALD) ---")
    for h in bald_history:
        stats = h.get("pool_stats", {})
        n = stats.get("labeled_normal", "?")
        p = stats.get("labeled_pneumonia", "?")
        total = h["n_labeled"]
        pct = f"{n}/{total}" if n != "?" else "N/A"
        print(f"  Round {h['round']}: {total} labeled ({pct} NORMAL)")

    print("\n" + "=" * 60)
    print("  NOTE: Full experiment uses 3 seeds x 300 budget x 10 MC passes")
    print("  Full results are pre-computed in experiments/results/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick demo for VIVA presentation")
    parser.add_argument(
        "--data_root", type=str, default="data/chest_xray",
        help="Path to chest_xray dataset root"
    )
    args = parser.parse_args()
    run_demo(args.data_root)
