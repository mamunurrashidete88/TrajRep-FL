

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import MetricsTracker, compute_eac


class FederatedEvaluator:
    
    def __init__(
        self,
        test_loader: DataLoader,
        backdoor_test_loader: Optional[DataLoader],
        num_classes: int,
        target_class: int = 0,
        target_acc_threshold: float = 60.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.test_loader = test_loader
        self.bd_loader = backdoor_test_loader
        self.num_classes = num_classes
        self.target_class = target_class
        self.target_acc_threshold = target_acc_threshold
        self.device = device


        self.random_asr = 100.0 / num_classes

    # ------------------------------------------------------------------
    # Core metric computations
    # ------------------------------------------------------------------

    def evaluate_accuracy(self, model: nn.Module) -> float:

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                out = model(batch_x)
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() == 3:  # (B, T, V) → (B*T, V)
                    B, T, V = out.shape
                    out = out.reshape(B * T, V)
                    batch_y = batch_y.reshape(B * T)
                preds = out.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.numel()
        return 100.0 * correct / max(total, 1)

    def evaluate_asr(self, model: nn.Module) -> float:

        if self.bd_loader is None:
            return 0.0
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_x, batch_y in self.bd_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                out = model(batch_x)
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() == 3:
                    B, T, V = out.shape
                    out = out.reshape(B * T, V)
                    batch_y = batch_y.reshape(B * T)
                preds = out.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.numel()
        return 100.0 * correct / max(total, 1)

    @staticmethod
    def compute_fpr(
        isolated_this_round: List[int],
        adversarial_ids: Set[int],
        benign_sampled_ids: List[int],
    ) -> float:

        if not benign_sampled_ids:
            return 0.0
        fp = sum(1 for cid in isolated_this_round if cid not in adversarial_ids)
        return 100.0 * fp / len(benign_sampled_ids)

    @staticmethod
    def compute_eac_metric(
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        trust_weights: Dict[int, torch.Tensor],
        adversarial_ids: Set[int],
        active_ids: List[int],
    ) -> float:

        return compute_eac(
            client_deltas, trust_weights, adversarial_ids, active_ids
        )

    # ------------------------------------------------------------------
    # Round-level evaluation
    # ------------------------------------------------------------------

    def evaluate_round(
        self,
        model: nn.Module,
        round_idx: int,
        round_log: Dict[str, Any],
        adversarial_ids: Set[int],
        sampled_ids: List[int],
        client_deltas: Optional[Dict] = None,
        eval_every: int = 5,
    ) -> Dict[str, Any]:
        
        metrics: Dict[str, Any] = {"round": round_idx}

        if round_idx % eval_every == 0 or round_idx == 1:
            metrics["acc"] = self.evaluate_accuracy(model)
            metrics["asr"] = self.evaluate_asr(model)
        else:
            metrics["acc"] = None
            metrics["asr"] = None


        isolated = round_log.get("isolated_this_round", [])
        benign_sampled = [c for c in sampled_ids if c not in adversarial_ids]
        metrics["fpr"] = self.compute_fpr(isolated, adversarial_ids, benign_sampled)


        trust_weights = round_log.get("trust_weights", {})
        active_ids = round_log.get("active_clients_list", sampled_ids)
        if client_deltas and trust_weights:
            metrics["eac"] = self.compute_eac_metric(
                client_deltas, trust_weights, adversarial_ids, active_ids
            )
        else:
            metrics["eac"] = None

        if trust_weights:
            tau_vals = [v.item() if torch.is_tensor(v) else float(v)
                        for v in trust_weights.values()]
            adv_tau = [tau_vals[i] for i, cid in enumerate(list(trust_weights.keys()))
                       if cid in adversarial_ids]
            ben_tau = [tau_vals[i] for i, cid in enumerate(list(trust_weights.keys()))
                       if cid not in adversarial_ids]
            metrics["mean_trust_adv"] = float(np.mean(adv_tau)) if adv_tau else None
            metrics["mean_trust_ben"] = float(np.mean(ben_tau)) if ben_tau else None
            metrics["trust_separation"] = (
                metrics["mean_trust_ben"] - metrics["mean_trust_adv"]
                if (metrics["mean_trust_ben"] is not None and metrics["mean_trust_adv"] is not None)
                else None
            )

        return metrics

    # ------------------------------------------------------------------
    #  summary report
    # ------------------------------------------------------------------

    def final_report(
        self,
        metrics_tracker: MetricsTracker,
        defense_name: str,
        dataset_name: str,
        attack_strategy: str,
    ) -> Dict[str, Any]:

        summary = metrics_tracker.summary()
        report = {
            "defense": defense_name,
            "dataset": dataset_name,
            "attack": attack_strategy,
            # Primary metrics
            "ACC (%)": round(summary["acc"], 1),
            "ASR (%)": round(summary["asr"], 1),
            "FPR (%)": round(summary["fpr"], 1) if summary["fpr"] is not None else "N/A",
            # Additional metrics
            "EAC": round(summary["eac"], 4) if summary["eac"] else "N/A",
            "Convergence (rounds)": summary["convergence_rounds"] or ">200",
            # Trust diagnostics
            "random_asr_baseline (%)": round(self.random_asr, 2),
        }
        return report

    # ------------------------------------------------------------------
    # Score analysis
    # ------------------------------------------------------------------

    @staticmethod
    def compute_score_separation(
        scores_benign: List[float],
        scores_adversarial: List[float],
    ) -> Dict[str, float]:

        if not scores_benign or not scores_adversarial:
            return {}
        b = np.array(scores_benign)
        a = np.array(scores_adversarial)
        return {
            "mean_benign_score": float(b.mean()),
            "std_benign_score": float(b.std()),
            "mean_adv_score": float(a.mean()),
            "std_adv_score": float(a.std()),
            "separation_gap": float(a.mean() - b.mean()),
            "separation_ratio": float(a.mean() / max(b.mean(), 1e-8)),
        }


# ------------------------------------------------------------------
# 
# ------------------------------------------------------------------

def results_to_latex_row(
    method: str,
    dataset_results: Dict[str, Dict[str, Any]],
    datasets: List[str] = ("femnist", "cifar10", "shakespeare"),
) -> str:

    row = method
    for ds in datasets:
        r = dataset_results.get(ds, {})
        acc = r.get("ACC (%)", "--")
        asr = r.get("ASR (%)", "--")
        fpr = r.get("FPR (%)", "N/A")
        row += f" & {acc} & {asr} & {fpr}"
    row += " \\\\"
    return row


def export_results_to_csv(
    all_results: Dict[str, Dict[str, Any]],
    output_path: str,
):

    import csv
    rows = []
    for key, result in all_results.items():
        rows.append(result)

    if not rows:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results exported to {output_path}")


def load_and_summarise_results(results_dir: str) -> Dict[str, Any]:

    summary: Dict[str, Any] = defaultdict(dict)
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        defense = data.get("defense", "unknown")
        cfg = data.get("config", {})
        dataset = cfg.get("dataset", {}).get("name", "unknown")
        s = data.get("summary", {})
        summary[defense][dataset] = {
            "ACC (%)": round(s.get("acc", 0), 1),
            "ASR (%)": round(s.get("asr", 0), 1),
            "FPR (%)": round(s.get("fpr", 0), 1),
            "EAC":     round(s.get("eac", 0), 4),
        }
    return dict(summary)


# Alias so run_experiment.py can import `Evaluator` directly
Evaluator = FederatedEvaluator
