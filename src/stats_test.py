"""
stats_test.py
=============
Paired t-test comparing active learning (BALD) vs random sampling
across multiple seeds at matched label-budget checkpoints.

Why a paired t-test?
- We run both strategies with the SAME 5 random seeds.
- At each checkpoint (e.g., 200 labels used), we pair the active and random
  accuracy from seed 1, seed 2, ..., seed 5.
- The paired test accounts for seed-to-seed variation (e.g., some seeds have
  easier initial labeled sets), giving a more powerful test than unpaired.

The null hypothesis at each checkpoint:
  H0: mean(active_accuracy) == mean(random_accuracy)
If p < 0.05, we reject H0 and conclude BALD significantly outperforms random.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def load_results(
    results_dir: str,
    strategy: str,
    seeds: List[int],
) -> Dict[int, List[Dict]]:
    """
    Load result histories for a given strategy across seeds.

    Parameters
    ----------
    results_dir : str
        Directory containing result JSON files.
    strategy : str
        'active_bald' or 'random'.
    seeds : list of int
        Seeds to load.

    Returns
    -------
    dict
        {seed: history_list}
    """
    results = {}
    for seed in seeds:
        if strategy == "random":
            filename = f"random_seed_{seed}.json"
        else:
            filename = f"active_bald_seed_{seed}.json"

        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping seed {seed}")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)
        results[seed] = data["history"]

    return results


def get_accuracy_at_checkpoint(
    history: List[Dict],
    checkpoint: int,
) -> Optional[float]:
    """
    Get accuracy at a specific label-count checkpoint.
    Returns the accuracy from the round closest to (but not exceeding) the checkpoint.
    """
    best_match = None
    for entry in history:
        if entry["n_labeled"] <= checkpoint:
            best_match = entry
        else:
            break

    if best_match is None:
        return None
    return best_match["accuracy"]


def run_paired_ttest(
    results_dir: str,
    seeds: List[int],
    checkpoints: List[int] = None,
) -> pd.DataFrame:
    """
    Run paired t-tests at label-budget checkpoints.

    Parameters
    ----------
    results_dir : str
        Directory with result JSON files.
    seeds : list of int
        Seeds used in the experiment.
    checkpoints : list of int
        Label counts at which to compare (e.g., [200, 300, 400]).

    Returns
    -------
    pd.DataFrame
        Table with columns: checkpoint, active_mean, active_std,
        random_mean, random_std, t_statistic, p_value, significant.
    """
    if checkpoints is None:
        checkpoints = [200, 300, 400, 500]

    active_results = load_results(results_dir, "active_bald", seeds)
    random_results = load_results(results_dir, "random", seeds)

    rows = []

    for cp in checkpoints:
        active_accs = []
        random_accs = []

        for seed in seeds:
            if seed in active_results and seed in random_results:
                a_acc = get_accuracy_at_checkpoint(active_results[seed], cp)
                r_acc = get_accuracy_at_checkpoint(random_results[seed], cp)
                if a_acc is not None and r_acc is not None:
                    active_accs.append(a_acc)
                    random_accs.append(r_acc)

        if len(active_accs) < 2:
            print(f"Checkpoint {cp}: insufficient data (only {len(active_accs)} pairs)")
            rows.append({
                "checkpoint": cp,
                "n_pairs": len(active_accs),
                "active_mean": np.mean(active_accs) if active_accs else None,
                "active_std": np.std(active_accs) if active_accs else None,
                "random_mean": np.mean(random_accs) if random_accs else None,
                "random_std": np.std(random_accs) if random_accs else None,
                "t_statistic": None,
                "p_value": None,
                "significant": None,
            })
            continue

        # Paired t-test (same seeds → paired samples)
        t_stat, p_value = stats.ttest_rel(active_accs, random_accs)

        rows.append({
            "checkpoint": cp,
            "n_pairs": len(active_accs),
            "active_mean": np.mean(active_accs),
            "active_std": np.std(active_accs, ddof=1),
            "random_mean": np.mean(random_accs),
            "random_std": np.std(random_accs, ddof=1),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": "Yes" if p_value < 0.05 else "No",
        })

    df = pd.DataFrame(rows)
    return df


def format_results_table(df: pd.DataFrame) -> str:
    """Format the t-test results as a readable table for the report."""
    lines = []
    lines.append("=" * 90)
    lines.append("PAIRED T-TEST: Active Learning (BALD) vs Random Sampling")
    lines.append("=" * 90)
    lines.append(f"{'Labels':>8} | {'Active (mean+/-std)':>20} | {'Random (mean+/-std)':>20} | "
                 f"{'t-stat':>8} | {'p-value':>8} | {'Sig?':>5}")
    lines.append("-" * 90)

    for _, row in df.iterrows():
        if row["t_statistic"] is not None:
            active_str = f"{row['active_mean']:.4f} +/- {row['active_std']:.4f}"
            random_str = f"{row['random_mean']:.4f} +/- {row['random_std']:.4f}"
            lines.append(
                f"{row['checkpoint']:>8} | {active_str:>20} | {random_str:>20} | "
                f"{row['t_statistic']:>8.3f} | {row['p_value']:>8.4f} | "
                f"{row['significant']:>5}"
            )
        else:
            lines.append(f"{row['checkpoint']:>8} | {'insufficient data':>20} |")

    lines.append("=" * 90)
    lines.append("Significance level: alpha = 0.05 (two-tailed paired t-test)")
    lines.append("H0: No difference in accuracy between BALD and random sampling")

    return "\n".join(lines)


def save_stats_report(
    results_dir: str,
    seeds: List[int],
    checkpoints: List[int] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the full statistical analysis and save/print results.

    Parameters
    ----------
    results_dir : str
        Directory with result JSON files.
    seeds : list of int
        Seeds used.
    checkpoints : list of int
        Label counts to test at.
    output_path : str or None
        Path to save CSV. If None, saves to results_dir/ttest_results.csv.

    Returns
    -------
    pd.DataFrame
        The results table.
    """
    df = run_paired_ttest(results_dir, seeds, checkpoints)

    # Print formatted table
    table_str = format_results_table(df)
    print(table_str)

    # Save CSV
    if output_path is None:
        output_path = os.path.join(results_dir, "ttest_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Also save the formatted table as text
    txt_path = output_path.replace(".csv", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"Formatted table saved to {txt_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run paired t-test analysis")
    parser.add_argument("--results_dir", type=str, default="experiments/results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[200, 300, 400, 500])
    args = parser.parse_args()

    save_stats_report(args.results_dir, args.seeds, args.checkpoints)
