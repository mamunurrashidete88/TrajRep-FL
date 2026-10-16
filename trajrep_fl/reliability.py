"""
trajrep_fl/reliability.py
=========================
Interaction-Aware Reliability Module.

"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionMLP(nn.Module):

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        clip_val: float = 5.0,
    ):
        super().__init__()
        self.clip_val = clip_val
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        
        r = self.net(u).squeeze(-1)
        return r.clamp(-self.clip_val, self.clip_val)

    @staticmethod
    def to_reliability(r_tilde: torch.Tensor) -> torch.Tensor:
      
        return torch.sigmoid(r_tilde)


# ---------------------------------------------------------------------------
# Ablation variants (no learned parameters — weight computation only)
# ---------------------------------------------------------------------------

def temporal_only_score(u: torch.Tensor) -> torch.Tensor:
    """Use TSC only (index 0).  Higher TSC → lower reliability."""
    tsc = u[..., 0].clamp(0.0, 1.0)
    return 1.0 - tsc                    


def peer_only_score(u: torch.Tensor) -> torch.Tensor:
    """Use PSC only (index 1)."""
    psc = u[..., 1].clamp(0.0, 1.0)
    return 1.0 - psc


def fixed_fusion_score(
    u: torch.Tensor,
    w_tsc: float = 0.5,
    w_psc: float = 0.5,
) -> torch.Tensor:
    """Linear combination of TSC and PSC."""
    tsc = u[..., 0].clamp(0.0, 1.0)
    psc = u[..., 1].clamp(0.0, 1.0)
    inconsistency = w_tsc * tsc + w_psc * psc
    return 1.0 - inconsistency.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Reliability state (per-client history for recovery)
# ---------------------------------------------------------------------------

class ReliabilityStateManager:

    def __init__(self, recovery_window: int = 5):
        self.recovery_window = recovery_window
        # q score history indexed by participation round sequence
        self._q_history: Dict[int, List[Tuple[int, float]]] = {}
        # Last aggregation weight (for smoothness term)
        self._last_alpha: Dict[int, float] = {}

    def update_q(self, client_id: int, round_idx: int, q: float):
        if client_id not in self._q_history:
            self._q_history[client_id] = []
        self._q_history[client_id].append((round_idx, q))
        # Keep at most 2*N_w entries to bound memory
        max_keep = max(2 * self.recovery_window, 20)
        if len(self._q_history[client_id]) > max_keep:
            self._q_history[client_id] = self._q_history[client_id][-max_keep:]

    def get_recent_q(self, client_id: int, n: Optional[int] = None) -> List[float]:
        """Return the q values from the last n participation rounds."""
        n = n or self.recovery_window
        hist = self._q_history.get(client_id, [])
        return [q for _, q in hist[-n:]]

    def update_alpha(self, client_id: int, alpha: float):
        self._last_alpha[client_id] = alpha

    def get_last_alpha(self, client_id: int) -> Optional[float]:
        return self._last_alpha.get(client_id)

    def reset(self, client_id: int):
        self._q_history.pop(client_id, None)
        self._last_alpha.pop(client_id, None)
