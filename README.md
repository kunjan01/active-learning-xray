# Cost-Optimal Active Learning for Image Classification

An end-to-end active learning pipeline that trains a chest X-ray classifier
(pneumonia vs normal) while labeling as few images as possible. Uses **BALD**
(Bayesian Active Learning by Disagreement) with MC Dropout and **class-balanced
acquisition** to select the most informative images for labeling, and compares
against random sampling to demonstrate labeling cost savings.

## Key Results

> Run the experiments to populate this section.

After running `experiments/run_comparison.py`, the main results will be:
- **Accuracy vs. Labels plot**: `experiments/results/accuracy_vs_labels.png`
- **Paired t-test table**: `experiments/results/ttest_results.csv`
- **Detailed analysis**: `notebooks/results_analysis.ipynb`

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

**Windows users**: If you encounter an OpenMP error, set this before running:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

### 2. Download Dataset

Download the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

**Option A** — Kaggle API (requires API token in `~/.kaggle/kaggle.json`):
```bash
python download_data.py
```

**Option B** — Manual download:
1. Download from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Extract to `data/chest_xray/`
3. Verify structure:
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3. Run Experiments

```bash
# Quick dry run (1 seed, 2 rounds, to verify pipeline works)
python experiments/run_comparison.py --data_root data/chest_xray --budget 150 --k 25 --seeds 42 --mc_passes 10

# Full experiment: 3 seeds x 2 strategies (BALD + Random)
python experiments/run_comparison.py --data_root data/chest_xray --seeds 42 123 456 --mc_passes 10

# Full experiment with more MC passes (GPU recommended)
python experiments/run_comparison.py --data_root data/chest_xray --seeds 42 123 456 789 1024
```

**Time estimates (CPU / GPU):**
- Dry run: ~1.5 hours / ~10 min
- 3-seed run (mc=10): ~8-10 hours / ~1 hour
- 5-seed run (mc=30): ~20-30 hours / ~2-4 hours

### 4. Analyze Results

```bash
jupyter notebook notebooks/results_analysis.ipynb
```

Or run the statistical test standalone:
```bash
python -m src.stats_test --results_dir experiments/results
```

## Project Structure

```
active-learning-xray/
├── data/                    # Dataset (gitignored)
├── src/
│   ├── dataset.py           # Dataset class, transforms, seed/pool/test splits
│   ├── model.py             # ResNet18 with dropout, freeze/unfreeze, training
│   ├── mc_dropout.py        # MC Dropout: N stochastic forward passes
│   ├── acquisition.py       # BALD and entropy scoring functions
│   ├── active_learning.py   # Main AL loop: train→score→select→label→repeat
│   ├── baseline.py          # Random sampling baseline (same loop, random selection)
│   ├── evaluate.py          # Test set evaluation (accuracy, precision, recall, F1)
│   └── stats_test.py        # Paired t-test across seeds at matched checkpoints
├── experiments/
│   ├── run_comparison.py    # Master runner: 5 seeds × 2 strategies + analysis
│   └── results/             # Output JSON/CSV/PNG files
├── notebooks/
│   └── results_analysis.ipynb  # Plots and tables for the report
├── download_data.py         # Dataset download script
├── requirements.txt
└── README.md
```

## Method

### Active Learning Loop

```
labeled = seed_set (100 images, class-balanced)
unlabeled = remaining training pool

while labeling_budget_remaining:
    train(model, labeled)                        # warm-start (Adam, 7 epochs)
    evaluate(model, fixed_test_set)              # held-out test set
    scores = mc_dropout(model, unlabeled, N=30)  # 30 stochastic forward passes
    scores = BALD(scores)                        # mutual information
    top_k = select_balanced(scores, k=25)        # k/2 per class (class-balanced)
    labeled += reveal_labels(top_k)              # oracle step
    unlabeled -= top_k
```

### Class-Balanced Acquisition

A naive BALD implementation selects the k images with the highest uncertainty scores.
On imbalanced datasets (like chest X-rays with 3:1 PNEUMONIA:NORMAL), this tends to
over-select the majority class, causing the model to become biased.

Our **class-balanced** variant selects the top k/2 most uncertain images from each
class, maintaining a more balanced labeled set throughout the active learning process.
This is critical for preventing accuracy degradation in later rounds.

### BALD (Bayesian Active Learning by Disagreement)

BALD estimates the mutual information between the model prediction and the
model parameters:

```
I[y; θ | x, D] = H[y|x, D] - E_θ[H[y|x, θ]]
               = entropy_of_mean - mean_of_entropies
```

This captures **epistemic uncertainty** (what the model doesn't know about its
parameters), not just **aleatoric uncertainty** (inherent label noise). An image
gets a high BALD score when different dropout masks disagree, even if the
average prediction looks confident.

### Statistical Validation

- **5 random seeds** control initial seed-set selection, model initialization,
  and augmentation randomness.
- **Paired t-test** (`scipy.stats.ttest_rel`) at matched label-budget checkpoints
  (200, 300, 400, 500 labels).
- The pairing accounts for seed-to-seed variation, giving a more powerful test.

## Course Mapping

### Optimization Techniques
- **Adam optimizer** with learning rate 1e-4 for efficient training
- **Transfer learning / fine-tuning**: freeze early layers, only train `layer4` + FC
- **Warm-start**: continue from previous weights each round instead of reinitializing

### Statistics III
- **Paired t-test**: `scipy.stats.ttest_rel` for comparing strategies across matched seeds
- **Confidence intervals**: mean ± std accuracy across 5 seeds at each checkpoint
- **Experimental design**: 5 seeds × 2 strategies with controlled randomness

### Deep Learning
- **Transfer learning**: ResNet18 pretrained on ImageNet, fine-tuned for chest X-rays
- **MC Dropout**: Bayesian approximation via stochastic forward passes at inference time
- **BALD**: Information-theoretic acquisition function using mutual information
- **Data augmentation**: Random crops, flips, rotations for the labeled training set

## Hardware Requirements

- **GPU recommended**: MC Dropout runs 30 forward passes per unlabeled image.
  - With GPU: ~2-4 hours for the full experiment
  - CPU only: ~12-24+ hours
- The code auto-detects CUDA and uses GPU if available.

## License

Academic project — for educational purposes.
