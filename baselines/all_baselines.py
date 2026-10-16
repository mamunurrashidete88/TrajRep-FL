"""
baselines/all_baselines.py
==========================
All baseline defense implementations used for comparison.

Stateless:
  - FedAvg          (McMahan et al., 2017)
  - TrimmedMean     (Yin et al., 2018)
  - Krum / MultiKrum (Blanchard et al., 2017)

Probabilistic / frequency-domain:
  - BayBFed         (Kumari et al., 2023)
  - FLIP            (Zhang et al., 2023)  [vision only]

Stateful / trajectory-aware:
  - FedRecover      (Cao et al., 2023)
  - FLDetector      (Zhang et al., 2022)
  - LongShortHistory (Gupta et al., 2022)

All implement the BaseServer interface:
    aggregate(client_deltas, sampled_client_ids, round_idx, **kwargs)
    → (updated_model, log_dict)
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from fl.server_base import BaseServer


# =======================================================================
#  Helper utilities
# =======================================================================

def _weighted_avg(
    global_model: nn.Module,
    client_deltas: Dict[int, Dict[str, torch.Tensor]],
    active_ids: List[int],
    weights: Dict[int, float],
    lr: float,
    device: torch.device,
) -> nn.Module:
    if not active_ids:
        return global_model
    total_w = sum(weights[cid] for cid in active_ids) + 1e-8
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            agg = torch.zeros_like(param.data)
            for cid in active_ids:
                if name in client_deltas[cid]:
                    agg += weights[cid] * client_deltas[cid][name].to(device)
            param.add_(lr * agg / total_w)
    return global_model


def _flat(delta: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in delta.values()]).float()


# =======================================================================
#  FedAvg
# =======================================================================

class FedAvgServer(BaseServer):

    def aggregate(
        self,
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        sampled_client_ids: List[int],
        round_idx: int,
        learning_rate: float = None,
        **kwargs,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        lr = learning_rate or self.learning_rate
        weights = {cid: 1.0 for cid in sampled_client_ids}
        self.global_model = _weighted_avg(
            self.global_model, client_deltas,
            sampled_client_ids, weights, lr, self.device
        )
        return self.global_model, {"defense": "fedavg"}


# =======================================================================
#  Trimmed Mean  (Yin et al., 2018)
# =======================================================================

class TrimmedMeanServer(BaseServer):
 
    def __init__(self, *args, trim_fraction: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        sampled_client_ids: List[int],
        round_idx: int,
        learning_rate: float = None,
        **kwargs,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        lr = learning_rate or self.learning_rate
        n = len(sampled_client_ids)
        k = max(1, int(self.trim_fraction * n))

        log = {"defense": "trimmed_mean", "active_clients": n}
        isolated = []

        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                stacked = torch.stack(
                    [client_deltas[cid][name].to(self.device)
                     for cid in sampled_client_ids
                     if name in client_deltas[cid]],
                    dim=0,
                )  # (N, *shape)
                flat_shape = stacked.shape[1:]
                stacked_flat = stacked.reshape(stacked.size(0), -1)   # (N, D)

                sorted_vals, _ = stacked_flat.sort(dim=0)
                trimmed = sorted_vals[k: n - k]  # (N - 2k, D)
                mean_delta = trimmed.mean(dim=0).reshape(flat_shape)
                param.add_(lr * mean_delta)

        log["isolated_this_round"] = isolated
        return self.global_model, log


# =======================================================================
#  Krum / Multi-Krum  (Blanchard et al., 2017)
# =======================================================================

class KrumServer(BaseServer):
    

    def __init__(self, *args, num_adversarial: int = 0, multi_k: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = num_adversarial
        self.multi_k = multi_k

    def aggregate(
        self,
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        sampled_client_ids: List[int],
        round_idx: int,
        learning_rate: float = None,
        **kwargs,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        lr = learning_rate or self.learning_rate
        n = len(sampled_client_ids)
        f = min(self.f, n // 2 - 1)

        flat_grads = {cid: _flat(client_deltas[cid]).to(self.device)
                      for cid in sampled_client_ids}

        # Pairwise squared distances
        ids = sampled_client_ids
        scores = {}
        for i, cid_i in enumerate(ids):
            dists = []
            for j, cid_j in enumerate(ids):
                if i == j:
                    continue
                dists.append((flat_grads[cid_i] - flat_grads[cid_j]).pow(2).sum().item())
            dists_sorted = sorted(dists)
            # Sum of n - f - 2 nearest neighbours
            scores[cid_i] = sum(dists_sorted[: max(n - f - 2, 1)])

        # Select top multi_k by lowest score
        selected = sorted(scores, key=lambda x: scores[x])[: self.multi_k]

        weights = {cid: 1.0 for cid in selected}
        self.global_model = _weighted_avg(
            self.global_model, client_deltas, selected, weights, lr, self.device
        )

        # Clients NOT selected are effectively "isolated" for FPR tracking
        not_selected = [cid for cid in sampled_client_ids if cid not in selected]
        return self.global_model, {
            "defense": "krum",
            "selected": selected,
            "isolated_this_round": not_selected,
        }


# =======================================================================
#  BayBFed  (Kumari et al., 2023)
# =======================================================================

class BayBFedServer(BaseServer):
    

    def __init__(self, *args, uncertainty_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty_scale = uncertainty_scale
        self._running_mean: Dict[int, torch.Tensor] = {}
        self._running_var: Dict[int, torch.Tensor] = {}
        self._round_count: Dict[int, int] = defaultdict(int)

    def _get_uncertainty(self, cid: int, flat: torch.Tensor) -> float:
        """Estimate uncertainty as deviation from running mean."""
        if cid not in self._running_mean:
            self._running_mean[cid] = flat.clone()
            self._running_var[cid] = torch.zeros_like(flat)
            return 1.0

        n = self._round_count[cid]
        delta = flat - self._running_mean[cid]
        self._running_mean[cid] = self._running_mean[cid] + delta / (n + 1)
        delta2 = flat - self._running_mean[cid]
        self._running_var[cid] = (n * self._running_var[cid] + delta * delta2) / (n + 1)

        uncertainty = self._running_var[cid].mean().item()
        return max(uncertainty, 1e-8)

    def aggregate(
        self,
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        sampled_client_ids: List[int],
        round_idx: int,
        learning_rate: float = None,
        **kwargs,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        lr = learning_rate or self.learning_rate
        weights = {}
        for cid in sampled_client_ids:
            flat = _flat(client_deltas[cid]).to(self.device)
            u = self._get_uncertainty(cid, flat)
            self._round_count[cid] += 1
            weights[cid] = 1.0 / (1.0 + self.uncertainty_scale * u)

        self.global_model = _weighted_avg(
            self.global_model, client_deltas,
            sampled_client_ids, weights, lr, self.device
        )
        return self.global_model, {"defense": "baybfed"}


# =======================================================================
#  FLIP  (Zhang et al., 2023) — vision only
# =======================================================================

class FLIPServer(BaseServer):
    

    def __init__(self, *args, freq_threshold: float = 2.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq_threshold = freq_threshold
        self._freq_baseline: Optional[torch.Tensor] = None
        self._baseline_count: int = 0

    def _get_spectrum(self, flat: torch.Tensor) -> torch.Tensor:
        """Compute FFT magnitude spectrum of a flat gradient vector."""
        fft = torch.fft.rfft(flat.float())
        return fft.abs()

    def aggregate(
        self,
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        sampled_client_ids: List[int],
        round_idx: int,
        learning_rate: float = None,
        **kwargs,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        lr = learning_rate or self.learning_rate

        spectra = {cid: self._get_spectrum(_flat(client_deltas[cid]).to(self.device))
                   for cid in sampled_client_ids}

        # Update running baseline
        mean_spectrum = torch.stack(list(spectra.values())).mean(dim=0)
        if self._freq_baseline is None:
            self._freq_baseline = mean_spectrum
        else:
            self._freq_baseline = 0.9 * self._freq_baseline + 0.1 * mean_spectrum
        self._baseline_count += 1

        # Filter clients whose spectrum deviates too much
        accepted, rejected = [], []
        for cid, spec in spectra.items():
            deviation = (spec - self._freq_baseline).abs().mean().item()
            baseline_norm = self._freq_baseline.abs().mean().item() + 1e-8
            rel_dev = deviation / baseline_norm
            if rel_dev < self.freq_threshold:
                accepted.append(cid)
            else:
                rejected.append(cid)

        if not accepted:
            accepted = sampled_client_ids

        weights = {cid: 1.0 for cid in accepted}
        self.global_model = _weighted_avg(
            self.global_model, client_deltas, accepted, weights, lr, self.device
        )
        return self.global_model, {
            "defense": "flip",
            "isolated_this_round": rejected,
            "active_clients": len(accepted),
        }


# =======================================================================
#  FedRecover  (Cao et al., 2023)
# =======================================================================

class FedRecoverServer(BaseServer):
    

    def __init__(self, *args, deviation_threshold: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.thr = deviation_threshold
        self._mean: Dict[int, torch.Tensor] = {}
        self._std: Dict[int, torch.Tensor] = {}
        self._counts: Dict[int, int] = defaultdict(int)

    def _update_profile(self, cid: int, flat: torch.Tensor) -> float:
        """Update client gradient profile; return z-score deviation."""
        if cid not in self._mean:
            self._mean[cid] = flat.clone()
            self._std[cid] = torch.ones_like(flat) * 1e-4
            self._counts[cid] = 1
            return 0.0

        n = self._counts[cid]
        old_mean = self._mean[cid]
        self._mean[cid] = old_mean + (flat - old_mean) / (n + 1)
        self._std[cid] = torch.sqrt(
            (n * self._std[cid].pow(2) + (flat - old_mean) * (flat - self._mean[cid])) / (n + 1)
        ).clamp(min=1e-6)
        self._counts[cid] += 1

        z_score = ((flat - self._mean[cid]).abs() / self._std[cid]).mean().item()
        return z_score

    def aggregate(
        self,
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        sampled_client_ids: List[int],
        round_idx: int,
        learning_rate: float = None,
        **kwargs,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        lr = learning_rate or self.learning_rate
        accepted, rejected = [], []

        for cid in sampled_client_ids:
            flat = _flat(client_deltas[cid]).to(self.device)
            z = self._update_profile(cid, flat)
            if z < self.thr:
                accepted.append(cid)
            else:
                rejected.append(cid)

        if not accepted:
            accepted = sampled_client_ids

        weights = {cid: 1.0 for cid in accepted}
        self.global_model = _weighted_avg(
            self.global_model, client_deltas, accepted, weights, lr, self.device
        )
        return self.global_model, {
            "defense": "fedrecover",
            "isolated_this_round": rejected,
            "active_clients": len(accepted),
        }


# =======================================================================
#  FLDetector  (Zhang et al., 2022)
# =======================================================================

class FLDetectorServer(BaseServer):
    

    def __init__(self, *args, cos_threshold: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.cos_threshold = cos_threshold
        self._prev_flat: Dict[int, torch.Tensor] = {}
        self._isolation_set: set = set()

    def _cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = a / (a.norm() + 1e-8)
        b = b / (b.norm() + 1e-8)
        return (a * b).sum().item()

    def aggregate(
        self,
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        sampled_client_ids: List[int],
        round_idx: int,
        learning_rate: float = None,
        **kwargs,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        lr = learning_rate or self.learning_rate
        accepted, rejected = [], []

        for cid in sampled_client_ids:
            if cid in self._isolation_set:
                rejected.append(cid)
                continue

            flat = _flat(client_deltas[cid]).to(self.device)

            if cid in self._prev_flat:
                cos = self._cosine_sim(flat, self._prev_flat[cid])
                if cos < self.cos_threshold:
                    self._isolation_set.add(cid)
                    rejected.append(cid)
                else:
                    accepted.append(cid)
            else:
                accepted.append(cid)   # first round: accept

            self._prev_flat[cid] = flat.clone()

        if not accepted:
            accepted = sampled_client_ids

        weights = {cid: 1.0 for cid in accepted}
        self.global_model = _weighted_avg(
            self.global_model, client_deltas, accepted, weights, lr, self.device
        )
        return self.global_model, {
            "defense": "fldetector",
            "isolated_this_round": rejected,
            "active_clients": len(accepted),
            "total_isolated": len(self._isolation_set),
        }


# =======================================================================
#  Long-Short History  (Gupta et al., 2022)
# =======================================================================

class LongShortHistoryServer(BaseServer):
    

    def __init__(
        self,
        *args,
        gamma_fast: float = 0.9,
        gamma_slow: float = 0.05,
        div_threshold: float = 1.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma_fast = gamma_fast
        self.gamma_slow = gamma_slow
        self.div_threshold = div_threshold
        self._ema_fast: Dict[int, torch.Tensor] = {}
        self._ema_slow: Dict[int, torch.Tensor] = {}
        self._isolation_set: set = set()

    def _update_ema(self, cid: int, flat: torch.Tensor) -> float:
        """
        Update both EMAs and return divergence of current update from slow EMA.

        We compare the incoming flat update to the slow EMA (long-term reference),
        not fast-to-slow.  This correctly detects abrupt reversals: when a large
        negative update arrives, the slow EMA (near the long-term mean) diverges
        from the current update in cosine space, triggering isolation.
        """
        if cid not in self._ema_fast:
            self._ema_fast[cid] = flat.clone()
            self._ema_slow[cid] = flat.clone()
            return 0.0

        # Compute divergence BEFORE updating slow EMA
        # d_c(flat, slow_prev) = 1 - cos(flat, slow), scaled to [0, 20]
        s_prev = self._ema_slow[cid]
        cos = (flat * s_prev).sum() / ((flat.norm() + 1e-8) * (s_prev.norm() + 1e-8))
        divergence = (1.0 - cos.item()) * 10.0

        # Update EMAs after computing divergence
        self._ema_fast[cid] = self.gamma_fast * flat + (1 - self.gamma_fast) * self._ema_fast[cid]
        self._ema_slow[cid] = self.gamma_slow * flat + (1 - self.gamma_slow) * self._ema_slow[cid]

        return divergence

    def aggregate(
        self,
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        sampled_client_ids: List[int],
        round_idx: int,
        learning_rate: float = None,
        **kwargs,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        lr = learning_rate or self.learning_rate
        accepted, rejected = [], []

        for cid in sampled_client_ids:
            if cid in self._isolation_set:
                rejected.append(cid)
                continue

            flat = _flat(client_deltas[cid]).to(self.device)
            div = self._update_ema(cid, flat)

            if div > self.div_threshold:
                self._isolation_set.add(cid)
                rejected.append(cid)
            else:
                accepted.append(cid)

        if not accepted:
            accepted = sampled_client_ids

        weights = {cid: 1.0 for cid in accepted}
        self.global_model = _weighted_avg(
            self.global_model, client_deltas, accepted, weights, lr, self.device
        )
        return self.global_model, {
            "defense": "long_short_history",
            "isolated_this_round": rejected,
            "active_clients": len(accepted),
            "total_isolated": len(self._isolation_set),
        }


# =======================================================================
#  TrajRep-FL server wrapper 
# =======================================================================

class TrajRepFLServerWrapper(BaseServer):
   

    def __init__(
        self,
        global_model: nn.Module,
        model_dim: int,
        trajrep_cfg: dict,
        num_clients: int,
        clients_per_round_fraction: float = 0.1,
        learning_rate: float = 0.01,
        device: torch.device = torch.device("cpu"),
        seed: int = 42,
        ablation_overrides: Optional[dict] = None,
    ):
        super().__init__(
            global_model, num_clients, clients_per_round_fraction,
            learning_rate, device, seed,
        )
        from trajrep_fl.server import TrajRepFLServer
        cfg = dict(trajrep_cfg)           # shallow copy — do not mutate caller's dict
        cfg["num_clients"] = num_clients
        if ablation_overrides:
            cfg.update(ablation_overrides)
        self.trajrep_server = TrajRepFLServer(
            model_dim=model_dim,
            cfg=cfg,
            device=device,
            global_model=global_model,
            learning_rate=learning_rate,
            num_clients=num_clients,
            clients_per_round_fraction=clients_per_round_fraction,
        )

    def aggregate(
        self,
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        sampled_client_ids: List[int],
        round_idx: int,
        learning_rate: float = None,
        is_adversarial: Optional[Dict[int, bool]] = None,
        **kwargs,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        lr = learning_rate or self.learning_rate
        self.global_model, log = self.trajrep_server.process_round(
            global_model=self.global_model,
            client_deltas=client_deltas,
            sampled_client_ids=sampled_client_ids,
            round_idx=round_idx,
            learning_rate=lr,
        )
        # Expose trust weights for EAC / FPR computation
        log["trust_weights"] = {}
        log["active_clients_list"] = sampled_client_ids
        return self.global_model, log


# =======================================================================
#  Defense registry
# =======================================================================

DEFENSE_REGISTRY = {
    "fedavg": FedAvgServer,
    "trimmed_mean": TrimmedMeanServer,
    "krum": KrumServer,
    "baybfed": BayBFedServer,
    "flip": FLIPServer,
    "fedrecover": FedRecoverServer,
    "fldetector": FLDetectorServer,
    "long_short_history": LongShortHistoryServer,
    "trajrep_fl": TrajRepFLServerWrapper,
}


def build_server(
    name: str,
    global_model: nn.Module,
    cfg: dict,
    device: torch.device,
    ablation_overrides: Optional[dict] = None,
):

    if name not in DEFENSE_REGISTRY:
        raise ValueError(f"Unknown defense: {name}. Available: {list(DEFENSE_REGISTRY.keys())}")

    fed_cfg = cfg.get("federated", cfg)
    num_clients = fed_cfg.get("num_clients", 500)
    frac = fed_cfg.get("clients_per_round_fraction", 0.1)
    lr = fed_cfg.get("learning_rate", 0.01)
    seed = cfg.get("experiment", {}).get("seed", 42)

    if name == "trajrep_fl":
        model_dim = sum(p.numel() for p in global_model.parameters())
        return TrajRepFLServerWrapper(
            global_model=global_model,
            model_dim=model_dim,
            trajrep_cfg=cfg.get("trajrep_fl", {}),
            num_clients=num_clients,
            clients_per_round_fraction=frac,
            learning_rate=lr,
            device=device,
            seed=seed,
            ablation_overrides=ablation_overrides or {},
        )

    cls = DEFENSE_REGISTRY[name]
    return cls(
        global_model=global_model,
        num_clients=num_clients,
        clients_per_round_fraction=frac,
        learning_rate=lr,
        device=device,
        seed=seed,
    )
