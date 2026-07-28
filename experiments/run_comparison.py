"""
experiments/run_comparison.py
=============================
Master experiment runner: runs both active learning (BALD) and random baseline
for 5 different random seeds, saves all results, and runs statistical analysis.

Usage:
    python experiments/run_comparison.py --data_root data/chest_xray

This is the single entry point for reproducing all experimental results.
"""

import argparse
import json
import os
import sys
import time
from datetime import timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.active_learning import run_active_learning
from src.baseline import run_random_baseline
from src.stats_test import save_stats_report


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_CHECKPOINTS = [200, 300, 400, 500]


def run_all_experiments(
    data_root: str,
    seeds: list,
    budget: int = 500,
    k: int = 25,
    epochs: int = 7,
    mc_passes: int = 30,
    results_dir: str = "experiments/results",
):
    """
    Run the full comparison experiment.

    Steps:
    1. For each seed, run BALD active learning.
    2. For each seed, run random baseline.
    3. Run paired t-test analysis.
    """
    os.makedirs(results_dir, exist_ok=True)

    total_runs = len(seeds) * 2  # active + random for each seed
    completed = 0
    start_time = time.time()

    # ── Phase 1: Active Learning (BALD) ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: Active Learning with BALD")
    print("=" * 70)

    active_histories = {}
    for seed in seeds:
        completed += 1
        print(f"\n[{completed}/{total_runs}] Active Learning -- seed={seed}")

        history = run_active_learning(
            data_root=data_root,
            seed=seed,
            acquisition_fn_name="bald",
            budget=budget,
            k=k,
            epochs_per_round=epochs,
            mc_passes=mc_passes,
            save_dir=results_dir,
        )
        active_histories[seed] = history

        elapsed = time.time() - start_time
        avg_per_run = elapsed / completed
        remaining = avg_per_run * (total_runs - completed)
        print(f"  Elapsed: {timedelta(seconds=int(elapsed))} | "
              f"Est. remaining: {timedelta(seconds=int(remaining))}")

    # ── Phase 2: Random Baseline ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: Random Sampling Baseline")
    print("=" * 70)

    random_histories = {}
    for seed in seeds:
        completed += 1
        print(f"\n[{completed}/{total_runs}] Random Baseline -- seed={seed}")

        history = run_random_baseline(
            data_root=data_root,
            seed=seed,
            budget=budget,
            k=k,
            epochs_per_round=epochs,
            save_dir=results_dir,
        )
        random_histories[seed] = history

        elapsed = time.time() - start_time
        avg_per_run = elapsed / completed
        remaining = avg_per_run * (total_runs - completed)
        print(f"  Elapsed: {timedelta(seconds=int(elapsed))} | "
              f"Est. remaining: {timedelta(seconds=int(remaining))}")

    # ── Phase 3: Statistical Analysis ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 3: Statistical Analysis")
    print("=" * 70)

    # Auto-compute checkpoints from budget (evenly spaced from 150 to budget)
    n_checkpoints = 4
    step = max((budget - 100) // n_checkpoints, 25)
    checkpoints = list(range(100 + step, budget + 1, step))
    if not checkpoints:
        checkpoints = [budget]
    print(f"  Checkpoints: {checkpoints}")

    df = save_stats_report(
        results_dir=results_dir,
        seeds=seeds,
        checkpoints=checkpoints,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {timedelta(seconds=int(total_time))}")
    print(f"  Results saved to: {results_dir}/")
    print(f"  Files generated:")
    for f in sorted(os.listdir(results_dir)):
        if f.endswith((".json", ".csv", ".txt")):
            fpath = os.path.join(results_dir, f)
            size = os.path.getsize(fpath)
            print(f"    {f} ({size:,} bytes)")
    print("=" * 70)

    return active_histories, random_histories, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run active learning vs random baseline comparison"
    )
    parser.add_argument(
        "--data_root", type=str, default="data/chest_xray",
        help="Path to chest_xray dataset root (containing train/, test/)"
    )
    parser.add_argument(
        "--budget", type=int, default=500,
        help="Total labeling budget (default: 500)"
    )
    parser.add_argument(
        "--k", type=int, default=25,
        help="Images to acquire per round (default: 25)"
    )
    parser.add_argument(
        "--epochs", type=int, default=7,
        help="Training epochs per round (default: 7)"
    )
    parser.add_argument(
        "--mc_passes", type=int, default=30,
        help="MC dropout forward passes (default: 30)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Random seeds (default: 42 123 456 789 1024)"
    )
    parser.add_argument(
        "--results_dir", type=str, default="experiments/results",
        help="Output directory for results (default: experiments/results)"
    )
    args = parser.parse_args()

    run_all_experiments(
        data_root=args.data_root,
        seeds=args.seeds,
        budget=args.budget,
        k=args.k,
        epochs=args.epochs,
        mc_passes=args.mc_passes,
        results_dir=args.results_dir,
    )
