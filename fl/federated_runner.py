"""
fl/federated_runner.py
======================

"""

from __future__ import annotations

import copy
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from tqdm import tqdm

from fl.client import FLClient, ShakespeareClient
from evaluation.metrics import MetricsTracker

logger = logging.getLogger("fl_runner")


class FederatedRunner:


    def __init__(
        self,
        global_model: nn.Module,
        server,                         # any BaseServer subclass
        clients: List[FLClient],
        adversarial_clients: List[int], # indices into `clients`
        attack,                         # any BaseAttack subclass
        test_loader,
        backdoor_test_loader,
        cfg: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
        target_acc_threshold: float = 60.0,
    ):
        self.global_model = global_model.to(device)
        self.server = server
        self.clients = clients
        self.adversarial_ids = set(adversarial_clients)
        self.attack = attack
        self.test_loader = test_loader
        self.backdoor_test_loader = backdoor_test_loader
        self.cfg = cfg
        self.device = device

        fed_cfg = cfg.get("federated", cfg)
        self.num_rounds = fed_cfg.get("num_rounds", 200)
        self.clients_per_round = max(
            1,
            int(len(clients) * fed_cfg.get("clients_per_round_fraction", 0.1)),
        )
        self.learning_rate = fed_cfg.get("server_learning_rate", 1.0)  
        self.dataset_name = cfg.get("dataset", {}).get("name", "unknown")

        self.metrics = MetricsTracker(target_acc_threshold=target_acc_threshold)
        self.round_logs: List[Dict[str, Any]] = []
        self.rng = random.Random(cfg.get("experiment", {}).get("seed", 42))

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self) -> MetricsTracker:
        logger.info(
            f"Starting FL: {self.num_rounds} rounds, "
            f"{len(self.clients)} clients, "
            f"{len(self.adversarial_ids)} adversarial "
            f"({100*len(self.adversarial_ids)/len(self.clients):.0f}%)"
        )

        for round_idx in tqdm(range(1, self.num_rounds + 1), desc="FL Rounds"):
            round_start = time.time()

            # ── 1. Sample clients ─────────────────────────────────────
            sampled_ids = self.rng.sample(
                range(len(self.clients)), self.clients_per_round
            )

            # ── 2. Collect client updates ─────────────────────────────
            client_deltas: Dict[int, Dict[str, torch.Tensor]] = {}
            is_adversarial: Dict[int, bool] = {}

            for cid in sampled_ids:
                client = self.clients[cid]
                is_adv = cid in self.adversarial_ids
                is_adversarial[cid] = is_adv

                # Local training
                poison_spec = self._local_poison_spec(round_idx) if is_adv else None
                delta = client.train(self.global_model, round_idx, poison_spec=poison_spec)

                # Apply attack if adversarial
                if is_adv and self.attack is not None:
                    delta = self.attack.poison(
                        delta=delta,
                        global_model=self.global_model,
                        client_id=cid,
                        round_idx=round_idx,
                        sampled_ids=sampled_ids,
                    )

                client_deltas[cid] = delta

            # ── 3. Server aggregation (with defense) ──────────────────
            self.global_model, agg_log = self.server.aggregate(
                client_deltas=client_deltas,
                sampled_client_ids=sampled_ids,
                round_idx=round_idx,
                is_adversarial=is_adversarial,
                harmful_client_ids=list(self.adversarial_ids),
                learning_rate=self.learning_rate,
            )

            if (self.attack is not None and hasattr(self.attack, 'set_feedback')
                    and self.cfg.get('adversary', {}).get('expose_own_reliability', False)):
                for cid, q in agg_log.get('q_scores', {}).items():
                    if cid in self.adversarial_ids:
                        self.attack.set_feedback(cid, float(q))

            # ── 4. Evaluate ───────────────────────────────────────────
            if round_idx % 5 == 0 or round_idx == self.num_rounds:
                acc = self._evaluate_clean()
                asr = self._evaluate_backdoor()
            else:
                acc, asr = None, None

            # ── 5. Track FPR if defense supports it ───────────────────
            fpr = self._compute_fpr(agg_log, is_adversarial, sampled_ids)
            eac = self._compute_eac(agg_log, is_adversarial, client_deltas, sampled_ids)

            # ── 6. Log ────────────────────────────────────────────────
            round_log = {
                "round": round_idx,
                "acc": acc,
                "asr": asr,
                "fpr": fpr,
                "eac": eac,
                "time": time.time() - round_start,
                **agg_log,
            }
            self.round_logs.append(round_log)
            self.metrics.update(round_log)

            if round_idx % 10 == 0:
                logger.info(
                    f"Round {round_idx:3d} | "
                    f"ACC={acc:.2f}% | ASR={asr:.2f}% | "
                    f"FPR={fpr:.2f}% | EAC={eac:.4f}"
                    if acc is not None
                    else f"Round {round_idx:3d} | training..."
                )

        return self.metrics

    def _local_poison_spec(self, round_idx):
        """Map attack strategy to on-device poisoning fraction for supported payloads.
        Other payloads remain update-space attacks, and do not claim data poisoning.
        """
        adv = self.cfg.get('adversary', {})
        payload = adv.get('payload', 'backdoor')
        if payload not in ('backdoor', 'backdoor_token', 'label_flip'):
            return None
        strategy = adv.get('strategy', 'static')
        if strategy == 'score_aware':
            strength = 1.0  # local trigger poisoning plus constrained update steering
        if strategy == 'delayed_shift':
            strength = float(round_idx >= int(adv.get('delay_rounds', 30)))
        elif strategy == 'gradual_drift':
            strength = min(1.0, max(0.0, round_idx / max(1, int(adv.get('drift_duration', 50)))))
        elif strategy == 'burst':
            benign = int(adv.get('burst_benign_window', 5))
            bad = int(adv.get('burst_malicious_window', 5))
            strength = float(((round_idx - 1) % max(1, benign + bad)) >= benign)
        else:
            strength = 1.0
        return dict(adv, strength=strength)

    # ------------------------------------------------------------------
    # Evaluation 
    # ------------------------------------------------------------------

    def _evaluate_clean(self) -> float:
        self.global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                out = self.global_model(batch_x)
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() == 3:   
                    B, T, V = out.shape
                    out = out.reshape(B * T, V)
                    batch_y = batch_y.reshape(B * T)
                pred = out.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
        return 100.0 * correct / max(total, 1)

    def _evaluate_backdoor(self) -> float:
        if self.backdoor_test_loader is None:
            return None
        self.global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_x, batch_y in self.backdoor_test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                out = self.global_model(batch_x)
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() == 3:
                    B, T, V = out.shape
                    out = out.reshape(B * T, V)
                    batch_y = batch_y.reshape(B * T)
                pred = out.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
        return 100.0 * correct / max(total, 1)

    def _compute_fpr(self, agg_log, is_adversarial, sampled_ids) -> float:
        """Post-transition monitored-state FPR on benign *valid* participants."""
        valid = agg_log.get('valid_clients', sampled_ids)
        benign = [cid for cid in valid if not is_adversarial.get(cid, False)]
        monitored = set(agg_log.get('monitored_client_ids', agg_log.get('isolated_this_round', [])))
        return 100.0 * sum(cid in monitored for cid in benign) / len(benign) if benign else 0.0

    def _compute_eac(self, agg_log, is_adversarial, client_deltas, sampled_ids) -> float:
        """Norm of weighted adversarial update, using actual normalized alpha."""
        if 'EHC' in agg_log:
            return float(agg_log['EHC'])
        weights = agg_log.get('alpha')
        if not weights:
            return 0.0
        terms = []
        for cid, a in weights.items():
            if is_adversarial.get(cid, False):
                flat = torch.cat([v.reshape(-1) for v in client_deltas[cid].values()]).float()
                terms.append(float(a) * flat)
        return torch.stack(terms).sum(dim=0).norm().item() if terms else 0.0
