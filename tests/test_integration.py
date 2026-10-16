"""
tests/test_integration.py
==========================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from trajrep_fl.server import TrajRepFLServer


# ── Minimal test model ────────────────────────────────────────────────

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


MODEL_DIM = sum(p.numel() for p in TinyMLP().parameters())   # 22


def make_server(extra_cfg=None) -> TrajRepFLServer:
    base_cfg = {
        "embedding_dim":     16,
        "projection_dim":    32,
        "gru_layers":        1,
        "clip_R":            5.0,
        "gamma":             0.9,
        "gamma_fast":        0.7,
        "gamma_slow":        0.1,
        "beta":              0.5,
        "mlp_hidden":        16,
        "epsilon_agg":       1e-8,
        "delta_init":        0.3,
        "delta_min":         0.1,
        "delta_max":         0.7,
        "delta_floor":       0.05,
        "T_wu":              2,
        "upsilon_S":         0.10,
        "upsilon_V":         0.05,
        "kappa_rec":         5.0,
        "theta_rec":         0.6,
        "recovery_window":   3,
        "learning_rate_traj": 1e-3,
        "weight_decay_traj":  1e-5,
        "lambda_smooth":     0.1,
        "lambda_ent":        0.01,
        "grad_clip":         5.0,
    }
    if extra_cfg:
        base_cfg.update(extra_cfg)

    model = TinyMLP()
    server = TrajRepFLServer(
        model_dim=MODEL_DIM,
        cfg=base_cfg,
        device=torch.device("cpu"),
        global_model=model,
        learning_rate=0.01,
        num_clients=20,
    )
    return server


def make_client_deltas(client_ids, model):
    """Fake client deltas: each client sends a random parameter delta."""
    deltas = {}
    for cid in client_ids:
        deltas[cid] = {
            name: torch.randn_like(p) * 0.1
            for name, p in model.named_parameters()
        }
    return deltas


# ── Tests ─────────────────────────────────────────────────────────────

class TestFullRound:
    def test_basic_round_no_crash(self):
        server = make_server()
        ids = list(range(8))
        deltas = make_client_deltas(ids, server.global_model)
        model_before = {n: p.data.clone() for n, p in server.global_model.named_parameters()}
        _, log = server.aggregate(deltas, ids, round_idx=0)

        # Model should have changed
        model_after = dict(server.global_model.named_parameters())
        assert any(
            not torch.allclose(model_before[n], model_after[n].data)
            for n in model_before
        ), "Global model was not updated"

        # Log should contain expected keys
        for key in ["round", "delta", "mean_q", "num_monitored"]:
            assert key in log, f"Missing log key: {key}"

    def test_multiple_rounds(self):
        server = make_server()
        ids = list(range(6))
        for r in range(5):
            deltas = make_client_deltas(ids, server.global_model)
            _, log = server.aggregate(deltas, ids, round_idx=r)
            assert log["round"] == r

    def test_ehc_logged_when_harmful_ids_provided(self):
        server = make_server()
        ids = list(range(6))
        harmful = [0, 1]
        deltas = make_client_deltas(ids, server.global_model)
        _, log = server.aggregate(
            deltas, ids, round_idx=0, harmful_client_ids=harmful
        )
        assert "EHC" in log
        assert log["EHC"] >= 0.0

    def test_intermittent_participation(self):
        """Clients that skip rounds should retain their hidden states."""
        server = make_server()
        all_ids = list(range(10))

        # Round 0: all participate
        deltas = make_client_deltas(all_ids, server.global_model)
        server.aggregate(deltas, all_ids, round_idx=0)

        # Round 1: only half participate
        half = all_ids[:5]
        deltas = make_client_deltas(half, server.global_model)
        server.aggregate(deltas, half, round_idx=1)

        # Clients 5–9 should still have their hidden states from round 0
        for cid in all_ids[5:]:
            assert server.client_states.get_hidden(cid) is not None


class TestAblations:
    @pytest.mark.parametrize("ablation_key", [
        "ablation_no_encoder",
        "ablation_ema_encoder",
        "ablation_no_multi_res",
        "ablation_no_interaction",
        "ablation_no_recovery",
        "ablation_temporal_only",
        "ablation_peer_only",
        "ablation_fixed_fusion",
    ])
    def test_ablation_no_crash(self, ablation_key):
        server = make_server({ablation_key: True})
        ids = list(range(6))
        deltas = make_client_deltas(ids, server.global_model)
        _, log = server.aggregate(deltas, ids, round_idx=0)
        assert "round" in log
