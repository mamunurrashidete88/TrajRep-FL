#!/usr/bin/env bash
# scripts/run_all_experiments.sh
#
# Usage:
#   bash scripts/run_all_experiments.sh
#
# Set DATASETS, SEEDS, or METHODS to restrict scope:
#   DATASETS="femnist" bash scripts/run_all_experiments.sh
#   SEEDS="11 22" bash scripts/run_all_experiments.sh

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATASETS="${DATASETS:-femnist cifar10 shakespeare}"
SEEDS="${SEEDS:-11 22 33 44 55}"
METHODS="${METHODS:-fedavg trimmed_mean krum baybfed fedrecover flip long_short fldetector trajrep_fl}"

echo "================================================================"
echo " TrajRep-FL — Full Experiment Suite"
echo " Datasets : $DATASETS"
echo " Seeds    : $SEEDS"
echo "================================================================"

# ── Static harmful behavior ─────────────────────────────────
echo -e "\n[1/5] Table 1 — Static harmful behavior"
for ds in $DATASETS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      python experiments/run_experiment.py \
        --dataset "$ds" --method "$method" \
        --strategy static --seed "$seed"
    done
  done
done
# Sequential execution avoids oversubscribing RAM/GPU; no background workers.

# ── Dynamic trajectories ───────────────────────────────────
echo -e "\n[2/5] Figure 1 — Dynamic trajectories"
for ds in $DATASETS; do
  for strategy in delayed_shift gradual_drift; do
    for seed in $SEEDS; do
      python experiments/run_experiment.py \
        --dataset "$ds" --method trajrep_fl \
        --strategy "$strategy" --seed "$seed"
      # Also run key baselines for comparison
      for bl in fedavg fldetector fedrecover; do
        python experiments/run_experiment.py \
          --dataset "$ds" --method "$bl" \
          --strategy "$strategy" --seed "$seed"
      done
    done
  done
done
# Sequential execution avoids oversubscribing RAM/GPU; no background workers.

# ── Component ablation ───────────────────────────────────────
echo -e "\n[3/5] Table 2 — Component ablation"
python experiments/run_ablation.py --dataset all --seeds $SEEDS

# ── Score-aware adaptive ───────────────────────────────────
echo -e "\n[4/5] Appendix — Score-aware adaptive adversary"
for ds in $DATASETS; do
  for seed in $SEEDS; do
    python experiments/run_experiment.py \
      --dataset "$ds" --method trajrep_fl \
      --strategy score_aware --seed "$seed"
    for bl in fldetector fedrecover long_short; do
      python experiments/run_experiment.py \
        --dataset "$ds" --method "$bl" \
        --strategy score_aware --seed "$seed"
    done
  done
done
# Sequential execution avoids oversubscribing RAM/GPU; no background workers.

# ── Hyperparameter sensitivity ───────────────────────────
echo -e "\n[5/5] Tables 3–4 — Hyperparameter sensitivity"
python experiments/run_sensitivity.py --dataset all --param all --seeds $SEEDS

echo -e "\n================================================================"
echo " All experiments complete. Results in results/"
echo "================================================================"
