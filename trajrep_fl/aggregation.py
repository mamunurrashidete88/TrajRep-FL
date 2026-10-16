"""Exactly normalized reliability-gated aggregation and model update."""
from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn

class ReliabilityGatedAggregation:
    def __init__(self, epsilon_agg: float = 0.0):
        # Legacy epsilon is ignored: positive sigmoid scores + clipped logits
        # ensure a positive denominator for every nonempty valid cohort.
        self.epsilon_agg = 0.0

    def compute_weights(self, client_ids: List[int], q_scores: Dict[int, float], r_tilde: Dict[int, float]) -> Dict[int, float]:
        if not client_ids:
            return {}
        q = torch.tensor([q_scores[i] for i in client_ids], dtype=torch.float64)
        r = torch.tensor([r_tilde[i] for i in client_ids], dtype=torch.float64)
        alpha = self.compute_weights_tensor(client_ids, q, r)
        return {cid: float(alpha[j]) for j, cid in enumerate(client_ids)}

    def compute_weights_tensor(self, client_ids: List[int], q_tensor: torch.Tensor, r_tilde_tensor: torch.Tensor) -> torch.Tensor:
        if not client_ids:
            return q_tensor.new_zeros((0,))
        if q_tensor.numel() != len(client_ids) or r_tilde_tensor.numel() != len(client_ids):
            raise ValueError('Score and client counts must match')
        if not torch.isfinite(q_tensor).all() or not ((q_tensor > 0) & (q_tensor < 1)).all():
            raise ValueError('Reliability scores must be finite and in (0,1)')
        logits = r_tilde_tensor + torch.log(q_tensor)
        return torch.softmax(logits, dim=0)

    def aggregate(self, global_model: nn.Module, client_deltas: Dict[int, Dict[str, torch.Tensor]], alpha: Dict[int, float], active_client_ids: List[int], learning_rate: float, device: torch.device) -> nn.Module:
        if not active_client_ids:
            return global_model
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                aggregate = torch.zeros_like(param)
                for cid in active_client_ids:
                    aggregate.add_(client_deltas[cid][name].to(device), alpha=alpha[cid])
                param.add_(aggregate, alpha=learning_rate)
        return global_model

    def compute_effective_harmful_contribution(self, client_deltas: Dict[int, Dict[str, torch.Tensor]], alpha: Dict[int, float], harmful_ids: List[int], device: torch.device) -> float:
        accumulator: Optional[torch.Tensor] = None
        for cid in harmful_ids:
            if cid not in client_deltas or cid not in alpha:
                continue
            flat = torch.cat([v.reshape(-1) for v in client_deltas[cid].values()]).to(device).float()
            weighted = alpha[cid] * flat
            accumulator = weighted if accumulator is None else accumulator + weighted
        return accumulator.norm().item() if accumulator is not None else 0.0
