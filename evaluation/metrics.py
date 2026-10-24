

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class MetricsTracker:

    def __init__(self, target_acc_threshold: float = 60.0):
        self.target_acc_threshold = target_acc_threshold
        self._history: Dict[str, List] = defaultdict(list)
        self._convergence_round: Optional[int] = None

    def update(self, round_log: Dict[str, Any]):
        round_idx = round_log.get("round", len(self._history["round"]) + 1)
        self._history["round"].append(round_idx)

        for key in ["acc", "asr", "fpr", "eac", "mean_trust", "mean_rds",
                    "newly_isolated", "reintegrated", "total_isolated",
                    "active_clients", "delta"]:
            val = round_log.get(key)
            if val is not None:
                self._history[key].append(float(val))

        acc = round_log.get("acc")
        if (
            acc is not None
            and self._convergence_round is None
            and acc >= self.target_acc_threshold
        ):
            self._convergence_round = round_idx

    # ------------------------------------------------------------------
    # Aggregate 
    # ------------------------------------------------------------------

    def final_acc(self) -> float:
        vals = [v for v in self._history["acc"] if v is not None]
        if not vals:
            return 0.0
        return float(np.mean(vals[-10:]))

    def final_asr(self) -> float:
        vals = [v for v in self._history["asr"] if v is not None]
        if not vals:
            return 0.0
        return float(np.mean(vals[-10:]))

    def mean_fpr(self) -> float:
        vals = [v for v in self._history["fpr"] if v is not None]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def mean_eac(self) -> float:
        vals = [v for v in self._history["eac"] if v is not None]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def convergence_time(self) -> Optional[int]:
        return self._convergence_round

    def summary(self) -> Dict[str, Any]:
        return {
            "acc": self.final_acc(),
            "asr": self.final_asr(),
            "fpr": self.mean_fpr(),
            "eac": self.mean_eac(),
            "convergence_rounds": self.convergence_time(),
            "total_rounds": len(self._history["round"]),
        }

    def get_history(self, key: str) -> List[float]:
        return self._history.get(key, [])

    def to_csv_rows(self) -> List[Dict[str, Any]]:
        keys = list(self._history.keys())
        n = len(self._history["round"])
        rows = []
        for i in range(n):
            row = {}
            for k in keys:
                vals = self._history[k]
                row[k] = vals[i] if i < len(vals) else None
            rows.append(row)
        return rows


def compute_eac(
    client_deltas: Dict[int, Dict[str, torch.Tensor]],
    trust_weights: Dict[int, torch.Tensor],
    adversarial_ids: set,
    active_ids: List[int],
    epsilon_agg: float = 1e-8,
) -> float:
  
    total_trust = sum(
        trust_weights.get(cid, torch.tensor(1.0)).item() for cid in active_ids
    ) + epsilon_agg

    adv_active = [cid for cid in active_ids if cid in adversarial_ids]
    if not adv_active:
        return 0.0

    adv_sum = None
    for cid in adv_active:
        tau = trust_weights.get(cid, torch.tensor(1.0))
        tau_val = tau.item() if torch.is_tensor(tau) else float(tau)
        flat = torch.cat([v.reshape(-1) for v in client_deltas[cid].values()]).float()
        weighted = tau_val * flat
        if adv_sum is None:
            adv_sum = weighted
        else:
            adv_sum = adv_sum + weighted

    if adv_sum is None:
        return 0.0

    return (adv_sum / total_trust).norm().item()
