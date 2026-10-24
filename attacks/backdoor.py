
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.base_attack import BaseAttack


# =======================================================================
#  Label-flipping payload
# =======================================================================

class LabelFlipAttack(BaseAttack):

    def __init__(
        self,
        adversarial_client_ids: List[int],
        source_class: int,
        target_class: int = 0,
        scale: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(adversarial_client_ids, target_class, device)
        self.source_class = source_class
        self.scale = scale

    def poison(
        self,
        delta: Dict[str, torch.Tensor],
        global_model: nn.Module,
        client_id: int,
        round_idx: int,
        sampled_ids: List[int],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        return {name: -v * self.scale for name, v in delta.items()}


# =======================================================================
#  Backdoor injection payload
# =======================================================================

class BackdoorAttack(BaseAttack):


    def __init__(
        self,
        adversarial_client_ids: List[int],
        target_class: int = 0,
        trigger_size: int = 3,
        trigger_position: str = "bottom_right",
        scale: float = 10.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(adversarial_client_ids, target_class, device)
        self.trigger_size = trigger_size
        self.trigger_position = trigger_position
        self.scale = scale

    def add_trigger_to_image(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:

        triggered = images.clone()
        _, C, H, W = triggered.shape
        ts = self.trigger_size

        if self.trigger_position == "bottom_right":
            triggered[:, :, H - ts:H, W - ts:W] = 1.0
        elif self.trigger_position == "top_left":
            triggered[:, :, :ts, :ts] = 1.0
        return triggered

    def poison(
        self,
        delta: Dict[str, torch.Tensor],
        global_model: nn.Module,
        client_id: int,
        round_idx: int,
        sampled_ids: List[int],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        n_benign = max(len(sampled_ids) - len(
            [c for c in sampled_ids if c in self.adversarial_client_ids]
        ), 1)
        effective_scale = self.scale * n_benign

        return {name: v * effective_scale for name, v in delta.items()}


class TokenBackdoorAttack(BaseAttack):

    def __init__(
        self,
        adversarial_client_ids: List[int],
        target_class: int = 0,
        trigger_length: int = 5,
        scale: float = 8.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(adversarial_client_ids, target_class, device)
        self.trigger_length = trigger_length
        self.scale = scale

    def poison(
        self,
        delta: Dict[str, torch.Tensor],
        global_model: nn.Module,
        client_id: int,
        round_idx: int,
        sampled_ids: List[int],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        n_benign = max(
            len(sampled_ids) - len(
                [c for c in sampled_ids if c in self.adversarial_client_ids]
            ), 1
        )
        return {name: v * self.scale * n_benign for name, v in delta.items()}
