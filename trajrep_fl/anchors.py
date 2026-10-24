
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = F.normalize(a.unsqueeze(0), dim=-1)
    b_n = F.normalize(b.unsqueeze(0), dim=-1)
    return 1.0 - (a_n * b_n).sum()


class MultiResolutionAnchorManager:
    

    def __init__(
        self,
        embedding_dim: int,
        gamma: float = 0.9,
        gamma_fast: float = 0.7,
        gamma_slow: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        assert gamma_fast > gamma_slow, "gamma_fast must be > gamma_slow"
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.gamma_fast = gamma_fast
        self.gamma_slow = gamma_slow
        self.device = device

        # Per-client anchor tensors (initialised on first participation)
        self._single: Dict[int, torch.Tensor] = {}
        self._fast: Dict[int, torch.Tensor] = {}
        self._slow: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Feature computation 
    # ------------------------------------------------------------------

    def compute_features(
        self,
        client_id: int,
        z: torch.Tensor,
    ) -> Tuple[float, float, float]:

        if client_id not in self._single:
            # First participation — no prior anchor; distances initialised to 0
            return 0.0, 0.0, 0.0

        z = z.to(self.device)
        d_single = _cosine_distance(z, self._single[client_id]).item()
        d_fast   = _cosine_distance(z, self._fast[client_id]).item()
        d_slow   = _cosine_distance(z, self._slow[client_id]).item()
        return d_single, d_fast, d_slow

    # ------------------------------------------------------------------
    # Anchor update 
    # ------------------------------------------------------------------

    def update(self, client_id: int, z: torch.Tensor):

        z = z.detach().to(self.device)

        if client_id not in self._single:
            # Initialise all anchors to z (first observation)
            self._single[client_id] = z.clone()
            self._fast[client_id]   = z.clone()
            self._slow[client_id]   = z.clone()
        else:
            self._single[client_id] = (
                self.gamma * z + (1 - self.gamma) * self._single[client_id]
            )
            self._fast[client_id] = (
                self.gamma_fast * z + (1 - self.gamma_fast) * self._fast[client_id]
            )
            self._slow[client_id] = (
                self.gamma_slow * z + (1 - self.gamma_slow) * self._slow[client_id]
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_anchors_tensor(
        self, client_id: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if client_id not in self._single:
            return None, None, None
        return (
            self._single[client_id].detach(),
            self._fast[client_id].detach(),
            self._slow[client_id].detach(),
        )

    def get_single_anchor(self, client_id: int) -> Optional[torch.Tensor]:
        return self._single.get(client_id)

    def has_anchor(self, client_id: int) -> bool:
        return client_id in self._single

    def reset(self, client_id: int):
        for d in (self._single, self._fast, self._slow):
            d.pop(client_id, None)
