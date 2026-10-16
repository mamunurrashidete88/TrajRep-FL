

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from attacks.base_attack import BaseAttack


class ScoreAwareAdaptiveAttack(BaseAttack):
    

    def __init__(
        self,
        adversarial_client_ids: List[int],
        score_fn: Optional[Callable] = None,
        stealth_budget: float = 0.15,    # 
        pgd_steps: int = 20,
        pgd_lr: float = 0.01,
        target_class: int = 0,
        damage_scale: float = 5.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(adversarial_client_ids, target_class, device)
        self.score_fn = score_fn          # injected by server at runtime
        self.zeta_bar = stealth_budget
        self.pgd_steps = pgd_steps
        self.pgd_lr = pgd_lr
        self.damage_scale = damage_scale
        self._previous_q: Dict[int, float] = {}

    def set_feedback(self, client_id: int, q: float):
        """Expose only this malicious client's previously observed reliability."""
        self._previous_q[int(client_id)] = float(q)

    def set_score_fn(self, score_fn: Callable):
        self.score_fn = score_fn

    def poison(
        self,
        delta: Dict[str, torch.Tensor],
        global_model: nn.Module,
        client_id: int,
        round_idx: int,
        sampled_ids: List[int],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
 
        flat_benign = self.get_flat_delta(delta).to(self.device)

        # A score-aware attack cannot observe the *current* round's cohort
        # score before submission.  Its available feedback is its own LAST
        # submitted score, if the experiment makes that feedback visible.
        if self.score_fn is None:
            prev_q = self._previous_q.get(client_id)
            # Before feedback is available, use a bounded non-adaptive warm-up.
            # Higher previous q allows more aggressive deviations. The budget
            # bounds the distance from the benign update in parameter space.
            adaptation = 0.5 if prev_q is None else max(0.05, min(1.0, prev_q))
            budget = self.zeta_bar * adaptation * flat_benign.norm().clamp_min(1e-12)
            direction = -flat_benign / flat_benign.norm().clamp_min(1e-12)
            candidate = flat_benign.clone()
            for _ in range(self.pgd_steps):
                candidate = candidate + self.pgd_lr * direction
                movement = candidate - flat_benign
                candidate = flat_benign + movement * min(1.0, float(budget / movement.norm().clamp_min(1e-12)))
            return self.set_flat_delta(delta, candidate.detach().cpu())

        # Optional explicit score oracle, supplied only in experiments whose
        # threat model permits it. PGD ascent against the alignment objective,
        # projected into an L2 ball around the honest update.
        base = flat_benign.detach()
        radius = self.zeta_bar * base.norm().clamp_min(1e-12)
        delta_adv = base.clone()
        for _ in range(self.pgd_steps):
            direction = -base / base.norm().clamp_min(1e-12)
            delta_adv = delta_adv + self.pgd_lr * direction
            movement = delta_adv - base
            delta_adv = base + movement * min(1.0, float(radius / movement.norm().clamp_min(1e-12)))
            score = self.score_fn(delta_adv.detach())
            observed = float(score.item()) if torch.is_tensor(score) else float(score)
            if observed > self.zeta_bar:
                delta_adv = base + 0.5 * (delta_adv - base)

        return self.set_flat_delta(delta, delta_adv.detach().cpu())
