
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from attacks.base_attack import BaseAttack


class DelayedShiftAttack(BaseAttack):

    def __init__(
        self,
        payload_attack: BaseAttack,
        adversarial_client_ids: List[int],
        delay_rounds: int = 30,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(adversarial_client_ids, target_class=0, device=device)
        self.payload = payload_attack
        self.T_delay = delay_rounds
        self._activated: Dict[int, bool] = {}

    def is_delaying(self, round_idx: int) -> bool:
        """True while the client is still in the benign delay phase."""
        return round_idx <= self.T_delay

    def poison(
        self,
        delta: Dict[str, torch.Tensor],
        global_model: nn.Module,
        client_id: int,
        round_idx: int,
        sampled_ids: List[int],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.is_delaying(round_idx):
            # Delay phase: submit benign delta unchanged
            return delta

        # Active phase: forward to payload attack
        if not self._activated.get(client_id, False):
            self._activated[client_id] = True

        return self.payload.poison(
            delta=delta,
            global_model=global_model,
            client_id=client_id,
            round_idx=round_idx,
            sampled_ids=sampled_ids,
            **kwargs,
        )

    def get_phase(self, round_idx: int) -> str:
        return "delaying" if self.is_delaying(round_idx) else "active"
