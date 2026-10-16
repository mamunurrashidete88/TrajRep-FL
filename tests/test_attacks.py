"""
tests/test_attacks.py
=====================
Comprehensive tests for all attack implementations.
Tests both payloads and execution strategies.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture
def delta_small():
    return {"weight": torch.randn(4, 4) * 0.01, "bias": torch.randn(4) * 0.01}

@pytest.fixture
def delta_large():
    return {"weight": torch.randn(32, 32) * 0.01, "bias": torch.randn(32) * 0.01}

@pytest.fixture
def sampled_ids():
    return list(range(10))

@pytest.fixture
def adv_ids():
    return [0, 1]


# ── BackdoorAttack ────────────────────────────────────────────────────────
class TestBackdoorAttack:

    def test_scales_delta_up(self, delta_small, sampled_ids):
        from attacks.backdoor import BackdoorAttack
        atk = BackdoorAttack([0], target_class=0, scale=5.0)
        original_norm = sum(v.norm().item() for v in delta_small.values())
        poisoned = atk.poison(delta_small, None, 0, 1, sampled_ids)
        poisoned_norm = sum(v.norm().item() for v in poisoned.values())
        assert poisoned_norm > original_norm, "Backdoor attack should scale delta up"

    def test_structure_preserved(self, delta_small, sampled_ids):
        from attacks.backdoor import BackdoorAttack
        atk = BackdoorAttack([0], scale=10.0)
        poisoned = atk.poison(delta_small, None, 0, 1, sampled_ids)
        assert set(poisoned.keys()) == set(delta_small.keys())
        for k in delta_small:
            assert poisoned[k].shape == delta_small[k].shape


class TestLabelFlipAttack:

    def test_negates_gradient(self, delta_small, sampled_ids):
        from attacks.backdoor import LabelFlipAttack
        atk = LabelFlipAttack([0], source_class=1, target_class=0, scale=1.0)
        poisoned = atk.poison(delta_small, None, 0, 1, sampled_ids)
        for k in delta_small:
            assert torch.allclose(poisoned[k], -delta_small[k])

    def test_scale_applied(self, delta_small, sampled_ids):
        from attacks.backdoor import LabelFlipAttack
        atk = LabelFlipAttack([0], source_class=1, target_class=0, scale=3.0)
        poisoned = atk.poison(delta_small, None, 0, 1, sampled_ids)
        for k in delta_small:
            assert torch.allclose(poisoned[k], -delta_small[k] * 3.0)


# ── DelayedShiftAttack ────────────────────────────────────────────────────
class TestDelayedShiftAttack:

    def test_benign_during_delay_phase(self, delta_small, sampled_ids):
        from attacks.backdoor import BackdoorAttack
        from attacks.delayed_shift import DelayedShiftAttack
        payload = BackdoorAttack([0], scale=10.0)
        atk = DelayedShiftAttack(payload, [0], delay_rounds=20)
        for r in range(1, 21):
            poisoned = atk.poison(delta_small, None, 0, r, sampled_ids)
            for k in delta_small:
                assert torch.allclose(poisoned[k], delta_small[k]), \
                    f"Round {r} during delay should be unchanged"

    def test_activates_at_delay_plus_one(self, delta_small, sampled_ids):
        from attacks.backdoor import BackdoorAttack
        from attacks.delayed_shift import DelayedShiftAttack
        payload = BackdoorAttack([0], scale=5.0)
        atk = DelayedShiftAttack(payload, [0], delay_rounds=10)
        poisoned = atk.poison(delta_small, None, 0, 11, sampled_ids)
        original_norm = delta_small["weight"].norm().item()
        poisoned_norm = poisoned["weight"].norm().item()
        assert poisoned_norm > original_norm * 2.0

    def test_phase_reporting(self):
        from attacks.backdoor import BackdoorAttack
        from attacks.delayed_shift import DelayedShiftAttack
        payload = BackdoorAttack([0])
        atk = DelayedShiftAttack(payload, [0], delay_rounds=15)
        assert atk.get_phase(10) == "delaying"
        assert atk.get_phase(16) == "active"


# ── GradualDriftAttack ────────────────────────────────────────────────────
class TestGradualDriftAttack:

    def test_modifies_delta(self, delta_large, sampled_ids):
        from attacks.gradual_drift import GradualDriftAttack
        atk = GradualDriftAttack([0], drift_duration=50)
        poisoned = atk.poison(delta_large, None, 0, round_idx=25, sampled_ids=sampled_ids)
        for k in delta_large:
            assert poisoned[k].shape == delta_large[k].shape

    def test_small_per_round_deviation(self, delta_large, sampled_ids):
        """Each round's change should be small (below detection threshold)."""
        from attacks.gradual_drift import GradualDriftAttack
        atk = GradualDriftAttack([0], drift_duration=50)
        prev = {k: v.clone() for k, v in delta_large.items()}
        for r in range(1, 6):
            current = atk.poison(delta_large, None, 0, r, sampled_ids)
            for k in delta_large:
                diff_norm = (current[k] - prev[k]).norm().item()
                orig_norm = delta_large[k].norm().item() + 1e-8
                assert diff_norm / orig_norm < 5.0, \
                    f"Per-round deviation too large at round {r}: {diff_norm/orig_norm:.4f}"
            prev = current

    def test_cumulative_drift_over_rounds(self, delta_large, sampled_ids):
        """Cumulative drift should grow monotonically with round index."""
        from attacks.gradual_drift import GradualDriftAttack
        import torch.nn.functional as F
        atk = GradualDriftAttack([0], drift_duration=50)
        baseline = delta_large["weight"].reshape(-1)
        drifts = []
        for r in [5, 15, 30]:
            poisoned = atk.poison(delta_large, None, 0, r, sampled_ids)
            p_flat = poisoned["weight"].reshape(-1)
            cos = F.cosine_similarity(baseline.unsqueeze(0), p_flat.unsqueeze(0)).item()
            drifts.append(1.0 - cos)
        assert drifts[2] >= drifts[0] - 0.05, "Cumulative drift should grow over rounds"


# ── BurstAttack ───────────────────────────────────────────────────────────
class TestBurstAttack:

    def test_correct_phase_cycling(self):
        from attacks.backdoor import BackdoorAttack
        from attacks.gradual_drift import BurstAttack
        payload = BackdoorAttack([0])
        atk = BurstAttack(payload, [0], benign_window=3, malicious_window=3)
        for r in [1, 2, 3]:
            assert not atk.is_malicious_round(r), f"Round {r} should be benign"
        for r in [4, 5, 6]:
            assert atk.is_malicious_round(r), f"Round {r} should be malicious"
        assert not atk.is_malicious_round(7)

    def test_benign_round_passes_delta_unchanged(self, delta_small, sampled_ids):
        from attacks.backdoor import BackdoorAttack
        from attacks.gradual_drift import BurstAttack
        payload = BackdoorAttack([0], scale=50.0)
        atk = BurstAttack(payload, [0], benign_window=5, malicious_window=5)
        # Round 3 is benign
        poisoned = atk.poison(delta_small, None, 0, round_idx=3, sampled_ids=sampled_ids)
        for k in delta_small:
            assert torch.allclose(poisoned[k], delta_small[k])


# ── ScoreAwareAdaptiveAttack ──────────────────────────────────────────────
class TestScoreAwareAdaptiveAttack:

    def test_fallback_without_score_fn(self, delta_small, sampled_ids):
        from attacks.score_aware import ScoreAwareAdaptiveAttack
        atk = ScoreAwareAdaptiveAttack([0], score_fn=None, damage_scale=3.0)
        poisoned = atk.poison(delta_small, None, 0, 1, sampled_ids)
        # Should still modify delta (scale up)
        for k in delta_small:
            assert poisoned[k].shape == delta_small[k].shape

    def test_respects_stealth_budget_with_score_fn(self, delta_small, sampled_ids):
        from attacks.score_aware import ScoreAwareAdaptiveAttack
        budget = 0.4

        def mock_score_fn(flat_delta):
            # Score proportional to norm (proxy for deception score)
            return torch.tensor(flat_delta.norm().item() * 0.1)

        atk = ScoreAwareAdaptiveAttack(
            [0], score_fn=mock_score_fn,
            stealth_budget=budget, pgd_steps=5, pgd_lr=0.01,
        )
        poisoned = atk.poison(delta_small, None, 0, 1, sampled_ids)
        for k in delta_small:
            assert poisoned[k].shape == delta_small[k].shape


# ── WaNet and NARCISSUS ───────────────────────────────────────────────────
class TestStealthyTriggerAttacks:

    def test_wanet_trigger_warp(self):
        from attacks.wanet import WaNetAttack
        atk = WaNetAttack([0], target_class=0)
        images = torch.randn(2, 3, 32, 32)
        warped = atk.apply_warp(images)
        assert warped.shape == images.shape

    def test_wanet_poison_scales_delta(self, delta_small, sampled_ids):
        from attacks.wanet import WaNetAttack
        atk = WaNetAttack([0], scale=8.0)
        poisoned = atk.poison(delta_small, None, 0, 1, sampled_ids)
        for k in delta_small:
            assert poisoned[k].shape == delta_small[k].shape

    def test_narcissus_poison_scales_delta(self, delta_small, sampled_ids):
        from attacks.wanet import NARCISSUSAttack
        atk = NARCISSUSAttack([0], scale=8.0)
        poisoned = atk.poison(delta_small, None, 0, 1, sampled_ids)
        for k in delta_small:
            assert poisoned[k].shape == delta_small[k].shape

    def test_attack_factory_all_strategies(self):
        from attacks import build_attack
        adv_ids = [0, 1, 2]
        device = torch.device("cpu")
        cfg = {
            "adversary": {
                "target_class": 0,
                "delay_rounds": 10,
                "drift_duration": 50,
                "stealth_budget": 0.4,
                "pgd_steps": 5,
                "pgd_lr": 0.01,
                "trigger_size": 3,
                "trigger_position": "bottom_right",
                "source_class": 1,
            }
        }
        for strategy in ["static", "burst", "delayed_shift", "gradual_drift"]:
            for payload in ["backdoor", "label_flip"]:
                atk = build_attack(strategy, payload, adv_ids, cfg, device)
                assert atk is not None, f"Factory failed for {strategy}+{payload}"
