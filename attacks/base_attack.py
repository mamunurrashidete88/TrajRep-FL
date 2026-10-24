
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class BaseAttack(ABC):

    def __init__(
        self,
        adversarial_client_ids: List[int],
        target_class: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        self.adversarial_client_ids = set(adversarial_client_ids)
        self.target_class = target_class
        self.device = device
        self.round_counter: Dict[int, int] = {}

    @abstractmethod
    def poison(
        self,
        delta: Dict[str, torch.Tensor],
        global_model: nn.Module,
        client_id: int,
        round_idx: int,
        sampled_ids: List[int],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        ...

    def is_adversarial(self, client_id: int) -> bool:
        return client_id in self.adversarial_client_ids

    def _get_round_count(self, client_id: int) -> int:
        return self.round_counter.get(client_id, 0)

    def _increment_round(self, client_id: int):
        self.round_counter[client_id] = self.round_counter.get(client_id, 0) + 1

    def scale_delta(
        self,
        delta: Dict[str, torch.Tensor],
        scale: float,
    ) -> Dict[str, torch.Tensor]:
        return {k: v * scale for k, v in delta.items()}

    def add_noise(
        self,
        delta: Dict[str, torch.Tensor],
        noise_std: float,
    ) -> Dict[str, torch.Tensor]:
        return {k: v + torch.randn_like(v) * noise_std for k, v in delta.items()}

    def get_flat_delta(self, delta: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.reshape(-1) for v in delta.values()]).float()

    def set_flat_delta(
        self,
        delta: Dict[str, torch.Tensor],
        flat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        new_delta = {}
        offset = 0
        for name, v in delta.items():
            numel = v.numel()
            new_delta[name] = flat[offset:offset + numel].reshape(v.shape)
            offset += numel
        return new_delta
