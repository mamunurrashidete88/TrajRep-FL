**"TrajRep-FL: Beyond Round-Local Aggregation: Learning Cross-Session Client Trajectories for Robust Federated Learning"**

---

## Overview

TrajRep-FL is a trajectory representation learning framework for robust federated optimization.
It encodes each client's update history into a compact latent trajectory, maintains
multi-resolution temporal anchors, and computes aggregation weights through an
interaction-aware reliability MLP.

**Key components:**
- **Trajectory Encoder** 
- **Multi-Resolution Anchors** 
- **Feature Vector** 
- **Interaction MLP** 
- **Reliability-Gated Aggregation** 
- **Adaptive Recovery** 

---

## Repository Structure

```
TrajRep-FL/
├── trajrep_fl/          # Core TrajRep-FL implementation
│   ├── encoder.py       #   Sparse projection + GRU encoder + decoder
│   ├── anchors.py       #   Multi-resolution trajectory anchors
│   ├── features.py      #   TSC, PSC, ΔMR, feature vector construction
│   ├── reliability.py   #   Interaction MLP + reliability scoring
│   ├── aggregation.py   #   Reliability-gated aggregation (Eq. 4)
│   ├── recovery.py      #   Adaptive threshold + influence recovery (Eq. 6–8)
│   ├── objective.py     #   L_traj = L_rec + λ_smooth·L_smooth + λ_ent·L_ent
│   └── server.py        #   Full per-round pipeline (Algorithm 1)
├── baselines/           # All comparison methods
├── attacks/             # Payloads × execution strategies
├── models/              # FEMNIST CNN, CIFAR-10 CNN, Shakespeare LSTM
├── data/                # Dataset loaders (LEAF / Dirichlet)
├── fl/                  # FL client, federated runner, base server
├── evaluation/          # Metrics: ACC, ASR, FPR, EHC, TSI, WV
├── experiments/         # Experiment runners
├── configs/             # YAML hyperparameter configs
└── scripts/             # Data preparation + shell scripts
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare datasets

```bash
# FEMNIST (LEAF benchmark)
python scripts/prepare_femnist.py --output_dir data/femnist

# CIFAR-10
python scripts/prepare_cifar10.py --output_dir data/cifar10

# Shakespeare (LEAF benchmark)
python scripts/prepare_shakespeare.py --output_dir data/shakespeare
```

### 3. Run TrajRep-FL

```bash
python experiments/run_experiment.py --dataset femnist --strategy static

python experiments/run_experiment.py --dataset femnist --strategy delayed_shift

python experiments/run_experiment.py --dataset cifar10 --strategy gradual_drift

python experiments/run_experiment.py --dataset femnist --strategy score_aware

python experiments/run_experiment.py --dataset femnist --method fldetector --strategy static
```

## Reproducibility

```bash
bash scripts/run_all_experiments.sh
```

---

## 📜 License & Citation

Provided for academic peer review.
