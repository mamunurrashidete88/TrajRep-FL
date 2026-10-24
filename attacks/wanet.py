
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from attacks.base_attack import BaseAttack


# =======================================================================
#  WaNet (elastic image warping backdoor)
# =======================================================================

class WaNetAttack(BaseAttack):

    def __init__(
        self,
        adversarial_client_ids: List[int],
        target_class: int = 0,
        warp_strength: float = 0.5,
        grid_size: int = 4,
        noise_mode: bool = True,
        scale: float = 8.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(adversarial_client_ids, target_class, device)
        self.warp_strength = warp_strength
        self.grid_size = grid_size
        self.noise_mode = noise_mode
        self.scale = scale

        torch.manual_seed(2021)
        self._identity_grid: Optional[torch.Tensor] = None
        self._noise_grid: Optional[torch.Tensor] = None

    def _get_warp_grids(self, img_size: int) -> tuple:
        if self._identity_grid is not None:
            return self._identity_grid, self._noise_grid

        # Identity grid
        ins = torch.arange(-1, 1, 2.0 / img_size, device=self.device)
        y_grid, x_grid = torch.meshgrid(ins, ins, indexing="ij")
        identity = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)  
        self._identity_grid = identity

        # Noise grid (low-frequency elastic field)
        gsize = self.grid_size
        noise = torch.randn(1, gsize, gsize, 2, device=self.device) * self.warp_strength
        noise = F.interpolate(
            noise.permute(0, 3, 1, 2),
            size=(img_size, img_size),
            mode="bicubic",
            align_corners=True,
        ).permute(0, 2, 3, 1)
        self._noise_grid = noise

        return self._identity_grid, self._noise_grid

    def apply_warp(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        _, _, H, W = images.shape
        identity, noise = self._get_warp_grids(H)

        grid = (identity + noise).clamp(-1, 1)
        grid = grid.expand(images.size(0), -1, -1, -1)
        return F.grid_sample(images, grid, mode="bilinear", align_corners=True)

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


# =======================================================================
#  NARCISSUS 
# =======================================================================

class NARCISSUSAttack(BaseAttack):
    

    def __init__(
        self,
        adversarial_client_ids: List[int],
        target_class: int = 0,
        noise_eps: float = 8.0 / 255.0,
        pgd_steps: int = 100,
        scale: float = 8.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(adversarial_client_ids, target_class, device)
        self.noise_eps = noise_eps
        self.pgd_steps = pgd_steps
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


# =======================================================================
#  Registry
# =======================================================================

STEALTHY_TRIGGER_ATTACKS = {
    "wanet": WaNetAttack,
    "narcissus": NARCISSUSAttack,
}
