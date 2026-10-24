

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import torch


class AdaptiveThresholdController:


    def __init__(
        self,
        delta_init: float = 0.3,
        delta_min: float = 0.1,
        delta_max: float = 0.7,
        delta_floor: float = 0.05,
        T_wu: int = 10,
        upsilon_S: float = 0.10,
        upsilon_V: float = 0.05,
    ):
        self.delta = delta_floor
        self.delta_init = delta_init
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.delta_floor = delta_floor
        self.T_wu = T_wu
        self.upsilon_S = upsilon_S
        self.upsilon_V = upsilon_V

        
        self._prev_tsc: Dict[int, float] = {}

    def get_threshold(self, round_idx: int) -> float:
        
        return self.delta

    def update(
        self,
        round_idx: int,
        tsc_current: Dict[int, float],
        q_current: Dict[int, float],
    ):
        
        
        shifts = []
        for cid, tsc_t in tsc_current.items():
            if cid in self._prev_tsc:
                shifts.append(abs(tsc_t - self._prev_tsc[cid]))
        S_tsc = float(sum(shifts) / max(len(shifts), 1)) if shifts else 0.0

        
        q_vals = list(q_current.values())
        if len(q_vals) > 0:
            mean_q = sum(q_vals) / len(q_vals)
            V_q = sum((q - mean_q) ** 2 for q in q_vals) / len(q_vals)
        else:
            V_q = 0.0

        
        delta_prop = (
            self.delta
            + self.upsilon_S * S_tsc
            - self.upsilon_V * V_q
        )
        delta_prop = max(self.delta_min, min(self.delta_max, delta_prop))

        
        omega = min(round_idx / max(self.T_wu, 1), 1.0)
        self.delta = self.delta_floor + omega * (delta_prop - self.delta_floor)

        
        for cid, tsc_t in tsc_current.items():
            self._prev_tsc[cid] = tsc_t


class InfluenceRecoveryManager:


    def __init__(
        self,
        kappa_rec: float = 5.0,
        theta_rec: float = 0.6,
        recovery_window: int = 5,
    ):
        self.kappa_rec = kappa_rec
        self.theta_rec = theta_rec
        self.recovery_window = recovery_window

        
        self._monitored: Set[int] = set()

        
        self._rho_history: Dict[int, deque] = {}

    # ------------------------------------------------------------------
    # Readiness computation
    # ------------------------------------------------------------------

    def _readiness(self, q: float, delta: float) -> float:
        
        return 1.0 / (1.0 + math.exp(-self.kappa_rec * (q - delta)))

    # ------------------------------------------------------------------
    # Per-round processing
    # ------------------------------------------------------------------

    def process_round(
        self,
        participating_ids: List[int],
        q_scores: Dict[int, float],
        delta: float,
    ) -> Tuple[Set[int], List[int], List[int]]:

        newly_flagged: List[int] = []
        recovered: List[int] = []

        for cid in participating_ids:
            q = q_scores.get(cid, 0.5)
            rho = self._readiness(q, delta)

            # Maintain readiness history
            if cid not in self._rho_history:
                self._rho_history[cid] = deque(maxlen=self.recovery_window)
            self._rho_history[cid].append(rho)

            currently_monitored = cid in self._monitored

            if currently_monitored:
                # Check recovery condition
                hist = list(self._rho_history[cid])
                if len(hist) > 0 and (sum(hist) / len(hist)) > self.theta_rec:
                    self._monitored.discard(cid)
                    recovered.append(cid)
            else:
                # Check if client should enter monitoring
                if q < delta:
                    self._monitored.add(cid)
                    newly_flagged.append(cid)

        return set(self._monitored), newly_flagged, recovered

    def is_monitored(self, client_id: int) -> bool:
        return client_id in self._monitored

    def monitored_clients(self) -> Set[int]:
        return set(self._monitored)

    def stats(self) -> dict:
        return {
            "num_monitored": len(self._monitored),
        }

    def reset(self, client_id: int):
        self._monitored.discard(client_id)
        self._rho_history.pop(client_id, None)
