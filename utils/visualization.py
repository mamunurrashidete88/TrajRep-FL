"""
utils/visualization.py
=======================


"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# plot settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DEFENSE_COLORS = {
    "trajrep_fl":           "#e63946",
    "fldetector":         "#457b9d",
    "fedrecover":         "#2a9d8f",
    "long_short_history": "#e9c46a",
    "flip":               "#f4a261",
    "baybfed":            "#a8dadc",
    "trimmed_mean":       "#6d6875",
    "krum":               "#b5838d",
    "fedavg":             "#adb5bd",
}

DEFENSE_LABELS = {
    "trajrep_fl":           "TrajRep-FL (Ours)",
    "fldetector":         "FLDetector",
    "fedrecover":         "FedRecover",
    "long_short_history": "Long-Short Hist.",
    "flip":               "FLIP",
    "baybfed":            "BayBFed",
    "trimmed_mean":       "Trimmed Mean",
    "krum":               "Krum",
    "fedavg":             "FedAvg",
}

ATTACK_LABELS = {
    "static":        "Static",
    "burst":         "Burst",
    "delayed_shift": "Delayed Shift",
    "gradual_drift": "Gradual Drift",
    "score_aware":   "Score-Aware",
}

DATASET_LABELS = {
    "femnist":     "FEMNIST",
    "cifar10":     "CIFAR-10",
    "shakespeare": "Shakespeare",
}


def _load_results(results_dir: str) -> Dict[str, Any]:
    """
    Load all results.json files from the results directory tree.

    """
    all_results = {}
    for root, _dirs, files in os.walk(results_dir):
        for fname in files:
            if fname != "results.json":
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            cfg = data.get("config", {})
            run_tag = data.get("run_tag", "")
            parts = run_tag.split("_")
            method  = parts[1] if len(parts) > 1 else "unknown"
            dataset = cfg.get("dataset", {}).get("name", "unknown")
            attack  = cfg.get("adversary", {}).get("strategy", "unknown")
            payload = cfg.get("adversary", {}).get("payload", "unknown")
            seed    = cfg.get("experiment", {}).get("seed", 0)

            key = f"{method}_{dataset}_{attack}_{payload}_seed{seed}"
            all_results[key] = data
    return all_results


def plot_acc_temporal_attacks(
    results: Dict,
    datasets: List[str],
    attacks: List[str],
    defenses: List[str],
    output_path: str = "figures/Fig1_ACC_Temporal_Attacks.pdf",
):
    n_datasets = len(datasets)
    n_attacks = len(attacks)
    fig, axes = plt.subplots(
        n_attacks, n_datasets, figsize=(4 * n_datasets, 3.5 * n_attacks), squeeze=False
    )

    for row, attack in enumerate(attacks):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            defense_accs = {}
            for defense in defenses:
                key = f"{defense}_{dataset}_{attack}_backdoor"
                if key in results:
                    summary = results[key].get("summary", {})
                    defense_accs[defense] = summary.get("acc", 0.0)

            bars = ax.bar(
                range(len(defenses)),
                [defense_accs.get(d, 0) for d in defenses],
                color=[DEFENSE_COLORS.get(d, "#888") for d in defenses],
                width=0.6,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xticks(range(len(defenses)))
            ax.set_xticklabels(
                [DEFENSE_LABELS.get(d, d) for d in defenses],
                rotation=45, ha="right", fontsize=8,
            )
            ax.set_ylabel("Test Accuracy (%)")
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
            ax.set_title(
                f"{DATASET_LABELS.get(dataset, dataset)}\n{ATTACK_LABELS.get(attack, attack)}"
            )
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_asr_temporal_attacks(
    results: Dict,
    datasets: List[str],
    attacks: List[str],
    defenses: List[str],
    output_path: str = "figures/Fig2_ASR_Temporal_Attacks.pdf",
):
    """Plot attack success rate grouped by defense and execution strategy."""
    n_datasets = len(datasets)
    n_attacks = len(attacks)
    fig, axes = plt.subplots(
        n_attacks, n_datasets, figsize=(4 * n_datasets, 3.5 * n_attacks), squeeze=False
    )

    for row, attack in enumerate(attacks):
        for col, dataset in enumerate(datasets):
            ax = axes[row][col]
            defense_asr = {}
            for defense in defenses:
                key = f"{defense}_{dataset}_{attack}_backdoor"
                if key in results:
                    summary = results[key].get("summary", {})
                    defense_asr[defense] = summary.get("asr", 0.0)

            ax.bar(
                range(len(defenses)),
                [defense_asr.get(d, 0) for d in defenses],
                color=[DEFENSE_COLORS.get(d, "#888") for d in defenses],
                width=0.6, edgecolor="white", linewidth=0.5,
            )
            ax.set_xticks(range(len(defenses)))
            ax.set_xticklabels(
                [DEFENSE_LABELS.get(d, d) for d in defenses],
                rotation=45, ha="right", fontsize=8,
            )
            ax.set_ylabel("ASR (%)")
            ax.set_ylim(0, 100)
            ax.set_title(
                f"{DATASET_LABELS.get(dataset, dataset)}\n{ATTACK_LABELS.get(attack, attack)}"
            )
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_convergence_time(
    results: Dict,
    datasets: List[str],
    defenses: List[str],
    adv_fractions: List[float] = [0.10, 0.20, 0.30],
    output_path: str = "figures/Fig_Convergence_TargetRounds.pdf",
):
    
    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 4), squeeze=False)

    x = np.arange(len(adv_fractions))
    width = 0.8 / len(defenses)

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        for di, defense in enumerate(defenses):
            conv_times = []
            for frac in adv_fractions:
                key = f"{defense}_{dataset}_delayed_shift_backdoor"
                if key in results:
                    summary = results[key].get("summary", {})
                    ct = summary.get("convergence_rounds")
                    conv_times.append(ct if ct is not None else 200)
                else:
                    conv_times.append(200)

            offset = (di - len(defenses) / 2) * width + width / 2
            ax.bar(
                x + offset, conv_times,
                width=width * 0.9,
                label=DEFENSE_LABELS.get(defense, defense),
                color=DEFENSE_COLORS.get(defense, "#888"),
                edgecolor="white",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(f*100)}%" for f in adv_fractions])
        ax.set_xlabel("Adversarial Fraction")
        ax.set_ylabel("Rounds to Target Accuracy")
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_ylim(0, 220)
        ax.axhline(200, color="red", linestyle="--", alpha=0.5, label="Budget")
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_trust_evolution(
    trust_histories: Dict[str, List[float]],
    output_path: str = "figures/Trust_Evolution_TrajRepFL_TwoPanel.pdf",
    datasets: List[str] = ["femnist", "cifar10"],
):

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4))
    if len(datasets) == 1:
        axes = [axes]

    client_types = {
        "benign":        ("#2a9d8f", "Benign",        "-"),
        "static":        ("#adb5bd", "Static Adv.",   "--"),
        "delayed_shift": ("#e63946", "Delayed Shift", "-."),
        "gradual_drift": ("#e9c46a", "Gradual Drift", ":"),
    }

    for ax, dataset in zip(axes, datasets):
        ds_histories = trust_histories.get(dataset, {})
        for ctype, (color, label, ls) in client_types.items():
            vals = ds_histories.get(ctype, [])
            if vals:
                rounds = list(range(1, len(vals) + 1))
                ax.plot(rounds, vals, color=color, label=label,
                        linestyle=ls, linewidth=1.8)

        # Warm-up annotation
        ax.axvline(10, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.text(11, 0.95, "end warm-up", fontsize=8, color="gray", va="top")

        # Delayed-shift activation
        ax.axvline(30, color="#e63946", linestyle=":", alpha=0.4, linewidth=1)
        ax.text(31, 0.85, "T_delay+1", fontsize=8, color="#e63946", va="top")

        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Trust Weight τ")
        ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_figures(results_dir: str = "results", output_dir: str = "figures"):
    """Load all results and generate all paper figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Walk subdirectories — results live in results/{run_tag}/results.json
    result_files = [
        f for _, _, fs in os.walk(results_dir) for f in fs if f == "results.json"
    ]
    if not os.path.isdir(results_dir) or not result_files:
        print(f"No results.json found under '{results_dir}/'. Run experiments first.")
        return

    results = _load_results(results_dir)
    if not results:
        print("No valid result JSON files found.")
        return

    defenses_all = [
        "fedavg", "trimmed_mean", "krum", "baybfed",
        "fedrecover", "flip", "fldetector", "long_short_history", "trajrep_fl"
    ]
    datasets = ["femnist", "cifar10", "shakespeare"]

    plot_acc_temporal_attacks(
        results, datasets, ["delayed_shift", "gradual_drift"], defenses_all,
        os.path.join(output_dir, "Fig1_ACC_Temporal_Attacks.pdf"),
    )
    plot_asr_temporal_attacks(
        results, datasets, ["delayed_shift", "gradual_drift"], defenses_all,
        os.path.join(output_dir, "Fig2_ASR_Temporal_Attacks.pdf"),
    )
    plot_convergence_time(
        results, ["femnist", "cifar10"],
        ["trajrep_fl", "fldetector", "fedrecover", "trimmed_mean", "fedavg"],
        output_path=os.path.join(output_dir, "Fig_Convergence_TargetRounds.pdf"),
    )
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()
    generate_all_figures(args.results_dir, args.output_dir)


def plot_vision_trigger_results(
    results: Dict,
    datasets: List[str],
    acc_output_path: str = "figures/Fig3A_ACC_Vision_Triggers.pdf",
    asr_output_path: str = "figures/Fig3B_ASR_Vision_Triggers.pdf",
):

    triggers = ["wanet", "narcissus"]
    trigger_labels = {"wanet": "WaNet", "narcissus": "NARCISSUS"}

    defenses = [
        "trajrep_fl", "fldetector", "fedrecover", "long_short_history",
        "flip", "baybfed", "trimmed_mean", "krum", "fedavg",
    ]

    for metric, ylabel, out_path in [
        ("acc", "Test Accuracy (%)", acc_output_path),
        ("asr", "ASR (%)",           asr_output_path),
    ]:
        n_ds = len(datasets)
        n_tr = len(triggers)
        fig, axes = plt.subplots(n_tr, n_ds,
                                 figsize=(4.5 * n_ds, 3.5 * n_tr),
                                 squeeze=False)

        for row, trigger in enumerate(triggers):
            for col, dataset in enumerate(datasets):
                ax = axes[row][col]
                vals = [
                    results.get(dataset, {})
                           .get(trigger, {})
                           .get(d, {})
                           .get(metric, 0)
                    for d in defenses
                ]
                ax.bar(
                    range(len(defenses)), vals,
                    color=[DEFENSE_COLORS.get(d, "#888") for d in defenses],
                    width=0.6, edgecolor="white", linewidth=0.5,
                )
                ax.set_xticks(range(len(defenses)))
                ax.set_xticklabels(
                    [DEFENSE_LABELS.get(d, d) for d in defenses],
                    rotation=45, ha="right", fontsize=7,
                )
                ax.set_ylabel(ylabel)
                ax.set_ylim(0, 100)
                ax.set_title(
                    f"{DATASET_LABELS.get(dataset, dataset)}\n"
                    f"{trigger_labels.get(trigger, trigger)}"
                )
                ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


def plot_score_aware_asr(
    results: Dict,
    datasets: List[str],
    defenses: List[str],
    output_path: str = "figures/Fig4_ASR_Score_Aware.pdf",
):

    n_datasets = len(datasets)
    fig, axes = plt.subplots(1, n_datasets,
                             figsize=(4.5 * n_datasets, 4),
                             squeeze=False)

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        vals = [
            results.get(dataset, {}).get(d, {}).get("asr", 0)
            for d in defenses
        ]
        bars = ax.bar(
            range(len(defenses)), vals,
            color=[DEFENSE_COLORS.get(d, "#888") for d in defenses],
            width=0.6, edgecolor="white", linewidth=0.5,
        )
        ax.set_xticks(range(len(defenses)))
        ax.set_xticklabels(
            [DEFENSE_LABELS.get(d, d) for d in defenses],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_ylabel("ASR (%)")
        ax.set_ylim(0, 100)
        ax.set_title(
            f"{DATASET_LABELS.get(dataset, dataset)}\n"
            "Score-Aware Adaptive Adversary"
        )
        ax.grid(axis="y", alpha=0.3)
        # Annotate TrajRep-FL bar
        if "trajrep_fl" in defenses:
            idx = defenses.index("trajrep_fl")
            ax.bar_label(bars, labels=[
                f"{v:.1f}" if i == idx else "" for i, v in enumerate(vals)
            ], fontsize=8, padding=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
