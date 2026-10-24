

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from attacks.base_attack import BaseAttack


def _fixed_adversarial_direction(flat_size: int, seed: int = 42) -> torch.Tensor:
    
    g = torch.Generator()
    g.manual_seed(seed)
    v = torch.randn(flat_size, generator=g)
    return v / (v.norm() + 1e-8)


class GradualDriftAttack(BaseAttack):
    

    def __init__(
        self,
        adversarial_client_ids: List[int],
        drift_duration: int = 50,
        t_start: int = 0,
        target_class: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(adversarial_client_ids, target_class, device)
        self.T_drift = drift_duration
        self.t_start = t_start
        self._adv_dir: Optional[torch.Tensor] = None

    def _get_direction(self, flat_size: int) -> torch.Tensor:
        
        if self._adv_dir is None or self._adv_dir.numel() != flat_size:
            self._adv_dir = _fixed_adversarial_direction(flat_size).to(self.device)
        return self._adv_dir

    def _lambda(self, round_idx: int) -> float:
        return float(
            min(max((round_idx - self.t_start) / max(self.T_drift, 1), 0.0), 1.0)
        )

    def poison(
        self,
        delta: Dict[str, torch.Tensor],
        global_model: nn.Module,
        client_id: int,
        round_idx: int,
        sampled_ids: List[int],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        lam = self._lambda(round_idx)

        
        if lam == 0.0:
            return delta

        flat_ben = self.get_flat_delta(delta).to(self.device)
        ben_norm = flat_ben.norm()

        adv_dir = self._get_direction(flat_ben.numel())
        flat_harm = adv_dir * ben_norm  

        flat_mixed = (1.0 - lam) * flat_ben + lam * flat_harm
        return self.set_flat_delta(delta, flat_mixed.cpu())

    def status(self, round_idx: int) -> str:
        lam = self._lambda(round_idx)
        return f"gradual_drift(lambda={lam:.3f})"


# ---------------------------------------------------------------------------
# Burst attack 
# ---------------------------------------------------------------------------

class BurstAttack(BaseAttack):
    

    def __init__(
        self,
        payload_attack: BaseAttack,
        adversarial_client_ids: List[int],
        benign_window: int = 5,
        malicious_window: int = 5,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(adversarial_client_ids, target_class=0, device=device)
        self.payload = payload_attack
        self.benign_window = benign_window
        self.malicious_window = malicious_window
        self.cycle_length = benign_window + malicious_window

    def is_malicious_round(self, round_idx: int) -> bool:
        return ((round_idx - 1) % self.cycle_length) >= self.benign_window

    def poison(
        self,
        delta: Dict[str, torch.Tensor],
        global_model: nn.Module,
        client_id: int,
        round_idx: int,
        sampled_ids: List[int],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if not self.is_malicious_round(round_idx):
            return delta
        return self.payload.poison(
            delta, global_model, client_id, round_idx, sampled_ids, **kwargs
        )
