"""
fl/server_base.py
=================

"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class BaseServer(ABC):


    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        clients_per_round_fraction: float = 0.1,
        learning_rate: float = 0.01,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
    ):
        self.global_model = copy.deepcopy(global_model).to(device)
        self.num_clients = num_clients
        self.clients_per_round = max(1, int(num_clients * clients_per_round_fraction))
        self.learning_rate = learning_rate
        self.device = device
        self.rng = random.Random(seed)
        self.round_logs: List[Dict[str, Any]] = []

    def sample_clients(self, round_idx: int) -> List[int]:
        return self.rng.sample(range(self.num_clients), self.clients_per_round)

    def get_model_copy(self) -> nn.Module:
        return copy.deepcopy(self.global_model)

    def broadcast_model(self) -> Dict[str, torch.Tensor]:
        return {
            name: param.data.clone()
            for name, param in self.global_model.named_parameters()
        }

    @abstractmethod
    def aggregate(
        self,
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        sampled_client_ids: List[int],
        round_idx: int,
        **kwargs,
    ) -> Tuple[nn.Module, Dict[str, Any]]:

        ...

    def apply_flat_delta(
        self,
        model: nn.Module,
        delta: Dict[str, torch.Tensor],
        lr: float = 1.0,
    ) -> nn.Module:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in delta:
                    param.add_(lr * delta[name].to(self.device))
        return model

    def get_flat_params(self, model: nn.Module) -> torch.Tensor:
        return torch.cat([p.data.reshape(-1) for p in model.parameters()])

    def set_flat_params(self, model: nn.Module, flat: torch.Tensor):
        offset = 0
        for p in model.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].reshape(p.shape))
            offset += numel
