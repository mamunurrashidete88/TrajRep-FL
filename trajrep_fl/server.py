
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fl.server_base import BaseServer
from trajrep_fl.encoder import (
    SparseRandomProjection,
    TrajectoryEncoder,
    TrajectoryDecoder,
    ClientStateManager,
)
from trajrep_fl.anchors import MultiResolutionAnchorManager
from trajrep_fl.features import (
    build_feature_vector,
    build_feature_vector_tensor,
    compute_latent_centroid,
    compute_round_mean,
)
from trajrep_fl.reliability import (
    InteractionMLP,
    ReliabilityStateManager,
    temporal_only_score,
    peer_only_score,
    fixed_fusion_score,
)
from trajrep_fl.aggregation import ReliabilityGatedAggregation
from trajrep_fl.recovery import (
    AdaptiveThresholdController,
    InfluenceRecoveryManager,
)
from trajrep_fl.objective import trajectory_loss, reliability_target

logger = logging.getLogger("trajrep_fl")


class TrajRepFLServer(BaseServer):


    def __init__(
        self,
        model_dim: int,
        cfg: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        
        super().__init__(device=device, **kwargs)
        self.model_dim = model_dim
        self.cfg = cfg

        d_x = cfg.get("projection_dim", 128)
        d_z = cfg.get("embedding_dim", 64)
        n_layers = cfg.get("gru_layers", 2)
        clip_R = cfg.get("clip_R", 5.0)

        # ── 1. Sparse random projection  ──────────────────────
        self.projection = SparseRandomProjection(
            model_dim=model_dim,
            projection_dim=d_x,
            seed=cfg.get("projection_seed", 42),
        ).to(device)
        for p in self.projection.parameters():
            p.requires_grad_(False)

        # ── 2. GRU Encoder ───────────────────────────────────────────
        self.encoder = TrajectoryEncoder(
            projection_dim=d_x,
            embedding_dim=d_z,
            num_layers=n_layers,
        ).to(device)

        # ── 3. Auxiliary Decoder ─────────────────────────────────────
        self.decoder = TrajectoryDecoder(
            embedding_dim=d_z,
            projection_dim=d_x,
        ).to(device)

        # ── 4. Interaction MLP ───────────────────────────────────────
        self.mlp = InteractionMLP(
            input_dim=5,
            hidden_dim=cfg.get("mlp_hidden", 64),
            clip_val=clip_R,
        ).to(device)

        # ── 5. Joint optimiser (encoder + decoder + MLP) ─────────────
        self.optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.mlp.parameters()),
            lr=cfg.get("learning_rate_traj", 1e-3),
            weight_decay=cfg.get("weight_decay_traj", 1e-5),
        )

        # ── 6. Client state manager ──────────────────────────────────
        self.client_states = ClientStateManager(
            embedding_dim=d_z,
            num_gru_layers=n_layers,
            device=device,
        )

        # ── 7. Multi-resolution anchors ──────────────────────────────
        self.anchors = MultiResolutionAnchorManager(
            embedding_dim=d_z,
            gamma=cfg.get("gamma", 0.9),
            gamma_fast=cfg.get("gamma_fast", 0.7),
            gamma_slow=cfg.get("gamma_slow", 0.1),
            device=device,
        )

        # ── 8. Reliability state manager ─────────────────────────────
        self.rel_state = ReliabilityStateManager(
            recovery_window=cfg.get("recovery_window", 5),
        )

        # ── 9. Reliability-gated aggregation ─────────────────────────
        self.aggregator = ReliabilityGatedAggregation(
            epsilon_agg=cfg.get("epsilon_agg", 1e-8),
        )

        # ── 10. Adaptive threshold + recovery ────────────────────────
        self.threshold_ctrl = AdaptiveThresholdController(
            delta_init=cfg.get("delta_init", 0.3),
            delta_min=cfg.get("delta_min", 0.1),
            delta_max=cfg.get("delta_max", 0.7),
            delta_floor=cfg.get("delta_floor", 0.05),
            T_wu=cfg.get("T_wu", 10),
            upsilon_S=cfg.get("upsilon_S", 0.10),
            upsilon_V=cfg.get("upsilon_V", 0.05),
        )
        self.recovery_mgr = InfluenceRecoveryManager(
            kappa_rec=cfg.get("kappa_rec", 5.0),
            theta_rec=cfg.get("theta_rec", 0.6),
            recovery_window=cfg.get("recovery_window", 5),
        )

        # ── Ablation flags ───────────────────────────────────────────
        self._abl = {
            "no_encoder":        cfg.get("ablation_no_encoder",        False),
            "ema_encoder":       cfg.get("ablation_ema_encoder",        False),
            "no_multi_res":      cfg.get("ablation_no_multi_res",       False),
            "no_interaction":    cfg.get("ablation_no_interaction",     False),
            "no_recovery":       cfg.get("ablation_no_recovery",        False),
            "temporal_only":     cfg.get("ablation_temporal_only",      False),
            "peer_only":         cfg.get("ablation_peer_only",          False),
            "fixed_fusion":      cfg.get("ablation_fixed_fusion",       False),
        }

        # Misc hyperparameters
        self.beta = cfg.get("beta", 0.5)
        self.lambda_smooth = cfg.get("lambda_smooth", 0.1)
        self.lambda_ent = cfg.get("lambda_ent", 0.01)
        self.lambda_rel = cfg.get("lambda_cal", cfg.get("lambda_rel", 1.0))
        self.epsilon_cal = cfg.get("epsilon_cal", 1e-4)
        self.grad_clip = cfg.get("grad_clip", 5.0)

    # ------------------------------------------------------------------
    # Main entry point 
    # ------------------------------------------------------------------

    def aggregate(
        self,
        client_deltas: Dict[int, Dict[str, torch.Tensor]],
        sampled_client_ids: List[int],
        round_idx: int,
        learning_rate: Optional[float] = None,
        harmful_client_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[nn.Module, Dict[str, Any]]:

        lr = self.learning_rate if learning_rate is None else learning_rate
        log: Dict[str, Any] = {"round": round_idx}


        param_spec = [(name, tuple(p.shape)) for name, p in self.global_model.named_parameters()]
        param_names = {name for name, _ in param_spec}
        flat_deltas: Dict[int, torch.Tensor] = {}
        invalid_ids: List[int] = []
        for cid in sampled_client_ids:
            delta_map = client_deltas.get(cid)
            if (delta_map is None or set(delta_map) != param_names or
                any(not torch.is_tensor(delta_map[name]) or tuple(delta_map[name].shape) != shape
                    or not bool(torch.isfinite(delta_map[name]).all())
                    for name, shape in param_spec)):
                invalid_ids.append(cid)
                continue
            flat_deltas[cid] = torch.cat([delta_map[name].detach().reshape(-1)
                for name, _ in param_spec]).float().to(self.device)
        log['invalid_clients'] = invalid_ids
        log['valid_clients'] = list(flat_deltas)
        if not flat_deltas:
            log.update(delta=self.threshold_ctrl.delta, mean_q=0.0, newly_flagged=0,
                       recovered=0, num_monitored=len(self.recovery_mgr.monitored_clients()),
                       alpha={}, q_scores={}, active_clients_list=[], EHC=0.0)
            return self.global_model, log

        # ── Step 2–4: Project, encode, decode ────────────────────────
        projected: Dict[int, torch.Tensor] = {}
        x_hat_map: Dict[int, torch.Tensor] = {}
        embeddings: Dict[int, torch.Tensor] = {}

        self.encoder.train()
        self.decoder.train()

        for cid in sampled_client_ids:
            if cid not in flat_deltas:
                continue
            delta = flat_deltas[cid]

 
            eps_norm = self.cfg.get("eps_norm", 1e-8)
            delta_norm = delta / (delta.norm() + eps_norm)

            with torch.no_grad():
                x = self.projection(delta_norm.unsqueeze(0)).squeeze(0)
            projected[cid] = x


            if self._abl["no_encoder"]:
                z = x[:self.cfg.get("embedding_dim", 64)].detach()
                embeddings[cid] = z
                self.client_states.set_embedding(cid, z)
                self.client_states.increment(cid, round_idx)
                continue


            if self._abl["ema_encoder"]:
                prev_z = self.client_states.get_embedding(cid)
                z = x[:self.cfg.get("embedding_dim", 64)]
                if prev_z is not None:
                    z = self.cfg.get("gamma", 0.9) * z + (
                        1 - self.cfg.get("gamma", 0.9)
                    ) * prev_z
                z = z.detach()
                embeddings[cid] = z
                self.client_states.set_embedding(cid, z)
                self.client_states.increment(cid, round_idx)
                continue

            # Standard GRU encode
            hidden = self.client_states.get_hidden(cid)

            # Forward through encoder (
            x_seq = x.unsqueeze(0).unsqueeze(0)  # 
            if hidden is not None:
                hidden = hidden.to(self.device)
            z, new_hidden = self.encoder(x_seq, hidden)
            z = z.squeeze(0)              # 

            # Decode
            x_hat = self.decoder(z.unsqueeze(0)).squeeze(0)
            x_hat_map[cid] = x_hat

            embeddings[cid] = z
            self.client_states.set_embedding(cid, z.detach())
            self.client_states.set_hidden(cid, new_hidden)
            self.client_states.increment(cid, round_idx)

        # ── Step 5: Compute features  ─────────────────

        active_protocol_ids = list(embeddings.keys())

        
        centroid = compute_latent_centroid(
            {cid: embeddings[cid].detach() for cid in active_protocol_ids},
            active_protocol_ids,
        ).to(self.device)
        round_mean = compute_round_mean(
            flat_deltas, active_protocol_ids
        ).to(self.device)

        u_map: Dict[int, torch.Tensor] = {}   
        tsc_map: Dict[int, float] = {}
        mr_map: Dict[int, float] = {}
        st_map: Dict[int, float] = {}

        for cid in sampled_client_ids:
            if cid not in embeddings:
                continue
            z = embeddings[cid]  # 

            a_single, a_fast, a_slow = self.anchors.get_anchors_tensor(cid)

            
            if self._abl["no_multi_res"]:
                a_fast = a_single
                a_slow = a_single

            u = build_feature_vector_tensor(
                z=z,
                flat_delta=flat_deltas.get(cid, torch.zeros(1, device=self.device)),
                anchor_single=a_single,
                anchor_fast=a_fast,
                anchor_slow=a_slow,
                centroid=centroid,
                round_mean_delta=round_mean,
                beta=self.beta,
            )
            u_map[cid] = u
            tsc_map[cid] = float(u[0].detach())
            mr_map[cid] = float(u[2].detach())
            # q-hat uses raw single-scale ΔST, not combined TSC (u[0]).
            st_map[cid] = (0.0 if a_single is None else float(
                (1.0 - F.cosine_similarity(z.detach(), a_single.to(z.device), dim=0, eps=1e-12))))


        active_ids = [cid for cid in sampled_client_ids if cid in u_map]

        r_tilde_map: Dict[int, float] = {}
        q_map: Dict[int, float] = {}
        r_tensor_list: List[torch.Tensor] = []
        q_tensor_list: List[torch.Tensor] = []

        self.mlp.train()
        use_mlp = not (
            self._abl["no_interaction"]
            or self._abl["temporal_only"]
            or self._abl["peer_only"]
            or self._abl["fixed_fusion"]
        )

        for cid in active_ids:
            u = u_map[cid].to(self.device)
            if self._abl["no_interaction"] or self._abl["temporal_only"]:
                q_t = temporal_only_score(u)
                r_t = torch.zeros((), device=self.device)
            elif self._abl["peer_only"]:
                q_t = peer_only_score(u)
                r_t = torch.zeros((), device=self.device)
            elif self._abl["fixed_fusion"]:
                q_t = fixed_fusion_score(u)
                r_t = torch.zeros((), device=self.device)
            else:
                r_t = self.mlp(u.unsqueeze(0)).squeeze(0)  # clipped logit, graph
                q_t = torch.sigmoid(r_t)


            if not use_mlp:
                q_t = q_t.clamp(min=1e-6, max=1-1e-6)
            r_tensor_list.append(r_t.reshape(()))
            q_tensor_list.append(q_t.reshape(()))
            r_tilde_map[cid] = float(r_t.detach())
            q_map[cid] = float(q_t.detach())
            self.rel_state.update_q(cid, round_idx, q_map[cid])

        r_tensor = (
            torch.stack(r_tensor_list) if r_tensor_list
            else torch.zeros(0, device=self.device)
        )
        q_tensor = (
            torch.stack(q_tensor_list) if q_tensor_list
            else torch.zeros(0, device=self.device)
        )

        # ── Step 8: Recovery monitoring ───────────────────────────────
        delta = self.threshold_ctrl.get_threshold(round_idx)

        if not self._abl["no_recovery"]:
            _, newly_flagged, recovered = self.recovery_mgr.process_round(
                active_ids, q_map, delta
            )
        else:
            # Monitoring-only ablation: 
            newly_flagged, recovered = [], []
            for cid in active_ids:
                if q_map.get(cid, 1.0) < delta:
                    self.recovery_mgr._monitored.add(cid)

        log.update({
            "newly_flagged": len(newly_flagged),
            "recovered": len(recovered),
            "num_monitored": len(self.recovery_mgr.monitored_clients()),
            "monitored_client_ids": [cid for cid in active_ids if self.recovery_mgr.is_monitored(cid)],
            "delta": delta,
            "mean_q": (
                sum(q_map.values()) / max(len(q_map), 1) if q_map else 0.0
            ),
        })


        alpha_tensor = self.aggregator.compute_weights_tensor(
            active_ids, q_tensor, r_tensor
        )
        alpha: Dict[int, float] = {
            cid: float(alpha_tensor[i].detach())
            for i, cid in enumerate(active_ids)
        }

        log['alpha'] = dict(alpha)
        log['q_scores'] = dict(q_map)
        log['active_clients_list'] = list(active_ids)
        log['trust_weights'] = {cid: torch.tensor(alpha[cid]) for cid in active_ids}

        prev_alpha: Dict[int, Optional[float]] = {}
        for cid in active_ids:
            prev_alpha[cid] = self.rel_state.get_last_alpha(cid)
            self.rel_state.update_alpha(cid, alpha.get(cid, 0.0))

        # ── Step 10: Global model update (no grad) ────────────────────
        self.global_model = self.aggregator.aggregate(
            global_model=self.global_model,
            client_deltas=client_deltas,
            alpha=alpha,
            active_client_ids=active_ids,
            learning_rate=lr,
            device=self.device,
        )


        for cid, z in embeddings.items():
            self.anchors.update(cid, z.detach())


        if not self._abl["no_encoder"]:

            if use_mlp and active_ids:
                u_matrix = torch.stack([u_map[cid].detach() for cid in active_ids])
                target_features = u_matrix.clone()
                target_features[:, 0] = torch.tensor([st_map[cid] for cid in active_ids],
                                                        device=u_matrix.device, dtype=u_matrix.dtype)
                q_target = reliability_target(target_features, epsilon_cal=self.epsilon_cal)
                log['q_targets'] = {cid: float(q_target[j]) for j, cid in enumerate(active_ids)}
                log['delta_st'] = dict(st_map)
                lambda_rel = self.lambda_rel
            else:
                q_target = q_tensor.detach()
                lambda_rel = 0.0

            loss = trajectory_loss(
                x_hat=x_hat_map,
                x_target={
                    cid: projected[cid].to(self.device) for cid in projected
                },
                client_ids=sampled_client_ids,
                alpha_current=alpha_tensor,
                alpha_client_ids=active_ids,
                alpha_previous=prev_alpha,
                q_pred=q_tensor,
                q_target=q_target,
                lambda_smooth=self.lambda_smooth,
                lambda_ent=self.lambda_ent,
                lambda_rel=lambda_rel,
            )
            if loss.requires_grad:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters())
                    + list(self.decoder.parameters())
                    + list(self.mlp.parameters()),
                    self.grad_clip,
                )
                self.optimizer.step()
                log["L_traj"] = float(loss.detach())

        # ── Step 12: Update threshold ─────────────────────────────────
        self.threshold_ctrl.update(round_idx, tsc_map, q_map)

        # ── Optional EHC logging ──────────────────────────────────────
        if harmful_client_ids:
            harm_in_round = [
                cid for cid in harmful_client_ids
                if cid in alpha
            ]
            ehc = self.aggregator.compute_effective_harmful_contribution(
                client_deltas, alpha, harm_in_round, self.device
            )
            log["EHC"] = ehc

        return self.global_model, log
