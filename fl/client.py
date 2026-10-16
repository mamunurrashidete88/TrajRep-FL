"""
fl/client.py
============

"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


class FLClient:


    def __init__(
        self,
        client_id: int,
        dataset: Any,
        num_classes: int,
        local_epochs: int = 5,
        local_batch_size: int = 32,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        device: torch.device = torch.device("cpu"),
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.num_classes = num_classes
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.device = device

        self.data_loader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

    def train(
        self,
        global_model: nn.Module,
        round_idx: int = 0,
        poison_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        optimizer = optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self.data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                if poison_spec and poison_spec.get('strength', 0.0) > 0:
                    attack_type = poison_spec.get('payload')
                    if attack_type in ('backdoor', 'label_flip'):
                        mask = torch.rand(len(batch_y), device=self.device) < (
                            poison_spec.get('poison_fraction', .2) * poison_spec['strength'])
                        if attack_type == 'backdoor' and mask.any():
                            batch_x = batch_x.clone()
                            batch_y = batch_y.clone()
                            ts = int(poison_spec.get('trigger_size', 3))
                            if poison_spec.get('trigger_position', 'bottom_right') == 'top_left':
                                batch_x[mask, :, :ts, :ts] = 1.0
                            else:
                                batch_x[mask, :, -ts:, -ts:] = 1.0
                            batch_y[mask] = int(poison_spec.get('target_class', 0))
                        elif attack_type == 'label_flip' and mask.any():
                            batch_y = batch_y.clone()
                            source = int(poison_spec.get('source_class', 1))
                            batch_y[mask & (batch_y == source)] = int(poison_spec.get('target_class', 0))

                # A final singleton batch breaks BatchNorm1d in the FEMNIST head.
                # Skip that remainder rather than crashing a long FL experiment.
                if batch_x.size(0) == 1 and any(isinstance(m, nn.BatchNorm1d) for m in local_model.modules()):
                    continue
                optimizer.zero_grad()
                output = local_model(batch_x)
                # Handle sequence models (Shakespeare LSTM returns (logits, hidden))
                if isinstance(output, tuple):
                    output = output[0]
                if output.dim() == 3:
                    B, T, C = output.shape
                    output = output.reshape(B * T, C)
                    batch_y = batch_y.reshape(B * T)

                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 10.0)
                optimizer.step()


        delta = {}
        with torch.no_grad():
            for name, param in local_model.named_parameters():
                global_param = dict(global_model.named_parameters())[name]
                delta[name] = (param - global_param.to(self.device)).cpu()

        return delta

    def get_data_size(self) -> int:
        return len(self.dataset)


class ShakespeareClient(FLClient):

    def train(
        self,
        global_model: nn.Module,
        round_idx: int = 0,
        poison_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        optimizer = optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.local_epochs):
            for batch_x, batch_y in self.data_loader:
                batch_x = batch_x.to(self.device)   # (B, T)
                batch_y = batch_y.to(self.device)   # (B, T)

                if poison_spec and poison_spec.get('payload') == 'backdoor_token' and poison_spec.get('strength', 0.0) > 0:
                    mask = torch.rand(len(batch_y), device=self.device) < (
                        poison_spec.get('poison_fraction', .2) * poison_spec['strength'])
                    if mask.any():
                        batch_x = batch_x.clone()
                        batch_y = batch_y.clone()
                        batch_x[mask, :int(poison_spec.get('trigger_length', 5))] = 1
                        batch_y[mask, :] = int(poison_spec.get('target_class', 0))

                optimizer.zero_grad()
                logits, _ = local_model(batch_x)    # (B, T, V)
                B, T, V = logits.shape
                loss = criterion(logits.reshape(B * T, V), batch_y.reshape(B * T))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 5.0)
                optimizer.step()

        delta = {}
        with torch.no_grad():
            for name, param in local_model.named_parameters():
                global_param = dict(global_model.named_parameters())[name]
                delta[name] = (param - global_param.to(self.device)).cpu()

        return delta
