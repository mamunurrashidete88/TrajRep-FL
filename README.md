**"TrajRep-FL: Beyond Round-Local Aggregation: Learning Cross-Session Client Trajectories for Robust Federated Learning"**

---

## Overview

TrajRep-FL is a trajectory representation learning framework for robust federated optimization. It encodes each client's update history into a compact latent trajectory, maintains
multi-resolution temporal anchors, and computes aggregation weights through an interaction-aware reliability MLP.

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
├── README.md
├── requirements.txt
├── setup.py
├── pytest.ini
│
├── trajrep_fl/                  # Core TrajRep-FL implementation
│   ├── encoder.py              # Sparse projection, GRU, decoder, client states
│   ├── anchors.py              # Single-scale and fast/slow EMA anchors
│   ├── features.py             # Temporal, peer-relative, geometry features
│   ├── reliability.py          # Interaction MLP and reliability scoring
│   ├── aggregation.py          # Reliability-gated aggregation and EAC
│   ├── recovery.py             # Adaptive threshold, monitoring-only recovery
│   ├── objective.py            # Reconstruction, calibration, smoothness, entropy
│   └── server.py               # Per-round server implementation
│
├── baselines/
│   └── all_baselines.py        # FedAvg, Trimmed Mean, Krum, BayBFed,
│                               # FLIP, FedRecover-labeled, FLDetector,
│                               # Long-Short History implementations
│
├── attacks/
│   ├── base_attack.py
│   ├── backdoor.py
│   ├── delayed_shift.py
│   ├── gradual_drift.py
│   ├── score_aware.py
│   ├── wanet.py
│   ├── label_flip.py
│   └── narcissus.py
│
├── models/
│   ├── cnn_femnist.py
│   ├── cnn_cifar10.py
│   └── lstm_shakespeare.py
│
├── data/
│   ├── loaders.py
│   ├── femnist.py
│   ├── cifar10.py
│   └── shakespeare.py
│
├── fl/
│   ├── client.py
│   ├── federated_runner.py
│   └── server_base.py
│
├── evaluation/
│   ├── metrics.py
│   └── evaluator.py
│
├── experiments/
│   ├── run_experiment.py
│   ├── run_overall_comparison.py
│   ├── run_ablation.py
│   └── run_sensitivity.py
│
├── configs/
│   ├── femnist_config.yaml
│   ├── cifar10_config.yaml
│   ├── shakespeare_config.yaml
│   └── smoke_femnist.yaml
│
├── scripts/
│   ├── prepare_femnist.py
│   ├── prepare_cifar10.py
│   ├── prepare_shakespeare.py
│   ├── run_all_experiments.sh
│   ├── run_quick_test.sh
│   └── smoke_test.py
│
├── tests/
│   ├── conftest.py
│   ├── test_alignment.py
│   ├── test_attacks.py
│   ├── test_baselines.py
│   ├── test_data_and_models.py
│   ├── test_integration.py
│   └── test_trajrep_fl.py
│
├── utils/
│   ├── logging_utils.py
│   ├── reproducibility.py




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
