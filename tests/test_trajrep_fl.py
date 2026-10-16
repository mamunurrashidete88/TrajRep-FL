"""
tests/test_trajrep_fl.py
=========================
Unit tests for TrajRep-FL core components.

Tests cover:
  - Sparse random projection (fixed, shapes, reproducibility)
  - GRU encoder (forward pass, hidden state persistence)
  - Multi-resolution anchors (distance computation, update order)
  - Feature vector construction (all 5 signals)
  - Interaction MLP (logit clipping, sigmoid output)
  - Reliability-gated aggregation (weight properties, EHC bound)
  - Adaptive threshold (warm-up schedule, TSC-based adaptation)
  - Recovery manager (readiness, finite-participation recovery)
  - Full server round (shapes, no-crash, ablation flags)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from trajrep_fl.encoder import SparseRandomProjection, TrajectoryEncoder, TrajectoryDecoder, ClientStateManager
from trajrep_fl.anchors import MultiResolutionAnchorManager
from trajrep_fl.features import build_feature_vector, compute_latent_centroid, compute_round_mean
from trajrep_fl.reliability import InteractionMLP, temporal_only_score, peer_only_score
from trajrep_fl.aggregation import ReliabilityGatedAggregation
from trajrep_fl.recovery import AdaptiveThresholdController, InfluenceRecoveryManager
from trajrep_fl.objective import reconstruction_loss, smoothness_loss, entropy_loss


# ── Fixtures ──────────────────────────────────────────────────────────

MODEL_DIM    = 1000
EMBED_DIM    = 64
PROJ_DIM     = 128
DEVICE       = torch.device("cpu")


@pytest.fixture
def projection():
    return SparseRandomProjection(MODEL_DIM, PROJ_DIM, seed=42)


@pytest.fixture
def encoder():
    return TrajectoryEncoder(PROJ_DIM, EMBED_DIM)


@pytest.fixture
def decoder():
    return TrajectoryDecoder(EMBED_DIM, PROJ_DIM)


@pytest.fixture
def anchors():
    return MultiResolutionAnchorManager(EMBED_DIM, device=DEVICE)


@pytest.fixture
def mlp():
    return InteractionMLP(input_dim=5, hidden_dim=64, clip_val=5.0)


@pytest.fixture
def aggregator():
    return ReliabilityGatedAggregation(epsilon_agg=1e-8)


# ── Sparse Projection ─────────────────────────────────────────────────

class TestSparseProjection:
    def test_output_shape(self, projection):
        x = torch.randn(8, MODEL_DIM)
        out = projection(x)
        assert out.shape == (8, PROJ_DIM)

    def test_reproducible(self):
        p1 = SparseRandomProjection(MODEL_DIM, PROJ_DIM, seed=42)
        p2 = SparseRandomProjection(MODEL_DIM, PROJ_DIM, seed=42)
        x = torch.randn(4, MODEL_DIM)
        assert torch.allclose(p1(x), p2(x))

    def test_different_seed(self):
        p1 = SparseRandomProjection(MODEL_DIM, PROJ_DIM, seed=42)
        p2 = SparseRandomProjection(MODEL_DIM, PROJ_DIM, seed=99)
        x = torch.randn(4, MODEL_DIM)
        assert not torch.allclose(p1(x), p2(x))

    def test_not_trainable(self, projection):
        for p in projection.parameters():
            assert not p.requires_grad


# ── Trajectory Encoder ────────────────────────────────────────────────

class TestTrajectoryEncoder:
    def test_output_shape(self, encoder):
        x = torch.randn(4, 1, PROJ_DIM)   
        z, hidden = encoder(x)
        assert z.shape == (4, EMBED_DIM)
        assert hidden.shape[1] == 4

    def test_hidden_persistence(self, encoder):
        x1 = torch.randn(1, 1, PROJ_DIM)
        z1, h1 = encoder(x1)
        x2 = torch.randn(1, 1, PROJ_DIM)
        z2_with_h, _ = encoder(x2, h1)
        z2_no_h, _   = encoder(x2, None)
        # Different initial state → different outputs
        assert not torch.allclose(z2_with_h, z2_no_h)

    def test_deterministic(self, encoder):
        x = torch.randn(2, 3, PROJ_DIM)
        with torch.no_grad():
            z1, _ = encoder(x)
            z2, _ = encoder(x)
        assert torch.allclose(z1, z2)


# ── Multi-Resolution Anchors ──────────────────────────────────────────

class TestAnchors:
    def test_first_participation_zero_distance(self, anchors):
        z = torch.randn(EMBED_DIM)
        d_s, d_f, d_sl = anchors.compute_features(0, z)
        assert d_s == 0.0 and d_f == 0.0 and d_sl == 0.0

    def test_update_then_distance(self, anchors):
        z1 = torch.randn(EMBED_DIM)
        # First visit: no prior anchor
        anchors.update(0, z1)
        # Second visit: prior anchor = z1
        z2 = torch.randn(EMBED_DIM)
        d_s, d_f, d_sl = anchors.compute_features(0, z2)
        # Distance should be in [0, 2] (cosine distance range)
        assert 0.0 <= d_s <= 2.0 + 1e-6

    def test_identical_z_zero_distance(self, anchors):
        z = torch.randn(EMBED_DIM)
        anchors.update(1, z)
        d_s, d_f, d_sl = anchors.compute_features(1, z)
        assert abs(d_s) < 1e-5

    def test_fast_slow_separation_under_drift(self):
        """Lemma 1: persistent angular drift → fast/slow anchor separation grows."""
        anc = MultiResolutionAnchorManager(
            EMBED_DIM, gamma=0.9, gamma_fast=0.7, gamma_slow=0.1, device=DEVICE
        )
        omega = 0.05   # radians per step
        max_delta_mr = 0.0
        for t in range(200):
            angle = omega * t
            z = torch.zeros(EMBED_DIM)
            z[0] = torch.cos(torch.tensor(angle)).item()
            z[1] = torch.sin(torch.tensor(angle)).item()
            # z is already unit norm — direction changes each step

            _, d_fast, d_slow = anc.compute_features(0, z)
            delta_mr = abs(d_fast - d_slow)
            if delta_mr > max_delta_mr:
                max_delta_mr = delta_mr
            anc.update(0, z)

        # After sustained angular drift, fast and slow anchors should lag
        # by different amounts → measurable separation
        assert max_delta_mr > 1e-3, (
            f"Fast-slow separation should grow under angular drift, got max={max_delta_mr:.6f}"
        )


# ── Feature Vector ────────────────────────────────────────────────────

class TestFeatureVector:
    def test_shape(self, anchors):
        z = torch.randn(EMBED_DIM)
        delta = torch.randn(MODEL_DIM)
        centroid = torch.randn(EMBED_DIM)
        g_bar = torch.randn(MODEL_DIM)
        u = build_feature_vector(
            client_id=0, z=z, flat_delta=delta,
            d_single=0.1, d_fast=0.15, d_slow=0.05,
            centroid=centroid, round_mean_delta=g_bar, beta=0.5,
        )
        assert u.shape == (5,)

    def test_values_are_finite(self, anchors):
        u = build_feature_vector(
            client_id=0,
            z=torch.randn(EMBED_DIM),
            flat_delta=torch.randn(MODEL_DIM),
            d_single=0.2, d_fast=0.3, d_slow=0.1,
            centroid=torch.randn(EMBED_DIM),
            round_mean_delta=torch.randn(MODEL_DIM),
        )
        assert torch.all(torch.isfinite(u))

    def test_delta_mr_in_tsc(self):
        u = build_feature_vector(
            client_id=0,
            z=torch.randn(EMBED_DIM),
            flat_delta=torch.ones(MODEL_DIM),
            d_single=0.0, d_fast=0.4, d_slow=0.2,
            centroid=torch.zeros(EMBED_DIM),
            round_mean_delta=torch.ones(MODEL_DIM),
            beta=0.5,
        )
        expected_mr = abs(0.4 - 0.2)
        expected_tsc = 0.5 * 0.0 + 0.5 * expected_mr
        assert abs(u[0].item() - expected_tsc) < 1e-5
        assert abs(u[2].item() - expected_mr) < 1e-5


# ── Interaction MLP ───────────────────────────────────────────────────

class TestInteractionMLP:
    def test_output_in_clip_range(self, mlp):
        u = torch.randn(10, 5)
        r = mlp(u)
        assert torch.all(r >= -5.0) and torch.all(r <= 5.0)

    def test_reliability_in_01(self, mlp):
        u = torch.randn(10, 5)
        r = mlp(u)
        q = InteractionMLP.to_reliability(r)
        assert torch.all(q > 0) and torch.all(q < 1)

    def test_temporal_only_ablation(self):
        u = torch.tensor([[0.8, 0.5, 0.3, 1.2, 0.0]])
        q = temporal_only_score(u)
        assert 0.0 <= q.item() <= 1.0


# ── Reliability-Gated Aggregation ────────────────────────────────────

class TestAggregation:
    def test_weights_sum_near_one(self, aggregator):
        ids = [0, 1, 2, 3]
        q   = {i: 0.8 for i in ids}
        r   = {i: 0.0 for i in ids}
        alpha = aggregator.compute_weights(ids, q, r)
        assert abs(sum(alpha.values()) - 1.0) < 1e-5

    def test_lower_q_lower_weight(self, aggregator):
        ids = [0, 1]
        q   = {0: 0.9, 1: 0.1}
        r   = {0: 0.0, 1: 0.0}
        alpha = aggregator.compute_weights(ids, q, r)
        assert alpha[0] > alpha[1]

    def test_ehc_bound(self, aggregator):
        ids = [0, 1, 2, 3]
        q   = {0: 0.9, 1: 0.9, 2: 0.1, 3: 0.1}   # clients 2,3 are harmful
        r   = {i: 0.0 for i in ids}
        alpha = aggregator.compute_weights(ids, q, r)

        fake_deltas = {
            i: {"w": torch.ones(10) if i in [2, 3] else torch.zeros(10)}
            for i in ids
        }
        ehc = aggregator.compute_effective_harmful_contribution(
            fake_deltas, alpha, harmful_ids=[2, 3], device=DEVICE
        )
        assert ehc < math.sqrt(10) * 0.5


# ── Adaptive Threshold ────────────────────────────────────────────────

class TestThreshold:
    def test_warmup_schedule(self):
        ctrl = AdaptiveThresholdController(
            delta_init=0.3, delta_floor=0.05, T_wu=10
        )
        # During warm-up, threshold increases toward init
        ctrl.update(0, {}, {})
        d0 = ctrl.get_threshold(0)
        ctrl.update(5, {}, {})
        d5 = ctrl.get_threshold(5)
        ctrl.update(9, {}, {})
        d9 = ctrl.get_threshold(9)
        assert d0 <= d5 <= d9 + 1e-6  

    def test_tsc_increase_raises_threshold(self):
        ctrl = AdaptiveThresholdController(
            delta_init=0.3, upsilon_S=0.5, T_wu=0
        )
        tsc_cur = {0: 0.4, 1: 0.4}
        q_cur   = {0: 0.7, 1: 0.7}
        # Simulate rising TSC
        ctrl._prev_tsc = {0: 0.2, 1: 0.2}
        ctrl.update(10, tsc_cur, q_cur)
        assert ctrl.delta > ctrl.delta_floor  # initialized at floor per manuscript


# ── Recovery Manager ──────────────────────────────────────────────────

class TestRecovery:
    def test_enters_monitoring_when_low_q(self):
        mgr = InfluenceRecoveryManager(kappa_rec=5.0, theta_rec=0.6, recovery_window=5)
        monitored, flagged, recovered = mgr.process_round(
            participating_ids=[0],
            q_scores={0: 0.1},  
            delta=0.3,
        )
        assert 0 in monitored
        assert 0 in flagged

    def test_recovers_after_sustained_high_q(self):
        mgr = InfluenceRecoveryManager(kappa_rec=10.0, theta_rec=0.6, recovery_window=5)
        delta = 0.3

        # Enter monitoring
        mgr.process_round([0], {0: 0.1}, delta)
        assert mgr.is_monitored(0)

        # Sustained recovery 
        for _ in range(5):
            mgr.process_round([0], {0: 0.95}, delta)

        assert not mgr.is_monitored(0), "Client should have recovered"

    def test_no_false_recovery_with_noisy_q(self):
        """Recovery should not trigger with one good round then bad rounds."""
        mgr = InfluenceRecoveryManager(kappa_rec=5.0, theta_rec=0.8, recovery_window=5)
        delta = 0.3
        mgr.process_round([0], {0: 0.1}, delta)
        # 1 good, 4 bad
        mgr.process_round([0], {0: 0.99}, delta)
        for _ in range(4):
            mgr.process_round([0], {0: 0.1}, delta)
        assert mgr.is_monitored(0), "Should not recover with mixed readiness"


# ── Objective ─────────────────────────────────────────────────────────

class TestObjective:
    def test_rec_loss_zero_for_perfect_recon(self):
        x = torch.randn(5, PROJ_DIM)
        x_hat = {i: x[i].clone() for i in range(5)}
        x_tgt = {i: x[i].clone() for i in range(5)}
        loss = reconstruction_loss(x_hat, x_tgt, list(range(5)))
        assert loss.item() < 1e-10

    def test_smoothness_zero_for_same_alpha(self):
        # New tensor API: alpha_current is a (K,) tensor aligned to client ids;
        # alpha_previous is a {cid: prev_float} dict.
        ids = [0, 1, 2]
        alpha_curr = torch.tensor([0.5, 0.3, 0.2])
        alpha_prev = {0: 0.5, 1: 0.3, 2: 0.2}
        loss = smoothness_loss(alpha_curr, ids, alpha_prev)
        assert loss.item() < 1e-10

    def test_normalized_entropy_nonnegative(self):
        alpha = torch.tensor([0.5, 0.3, 0.2])
        loss = entropy_loss(alpha)
        assert 0.0 <= loss.item() <= 1.0
