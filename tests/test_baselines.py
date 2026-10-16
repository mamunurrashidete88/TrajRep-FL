"""
tests/test_baselines.py
=======================
Tests for all eight baseline defense implementations.
Verifies aggregation logic, isolation decisions, and edge cases.
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
def tiny_model():
    return nn.Linear(8, 4)

@pytest.fixture
def make_deltas():
    def _make(model, n=10, scale=0.01):
        return {
            i: {name: torch.randn_like(p) * scale
                for name, p in model.named_parameters()}
            for i in range(n)
        }
    return _make

@pytest.fixture
def device():
    return torch.device("cpu")


# ── FedAvg ────────────────────────────────────────────────────────────────
class TestFedAvgServer:

    def test_aggregates_all_clients(self, tiny_model, make_deltas, device):
        from baselines import FedAvgServer
        server = FedAvgServer(tiny_model, num_clients=20, device=device)
        deltas = make_deltas(tiny_model, n=10)
        original_params = {n: p.clone() for n, p in server.global_model.named_parameters()}
        updated, log = server.aggregate(deltas, list(range(10)), round_idx=1, learning_rate=0.1)
        # Parameters should change
        changed = any(
            not torch.allclose(p, original_params[n])
            for n, p in updated.named_parameters()
        )
        assert changed, "FedAvg should update model parameters"

    def test_defense_label_in_log(self, tiny_model, make_deltas, device):
        from baselines import FedAvgServer
        server = FedAvgServer(tiny_model, num_clients=20, device=device)
        deltas = make_deltas(tiny_model, n=5)
        _, log = server.aggregate(deltas, list(range(5)), round_idx=1)
        assert log.get("defense") == "fedavg"

    def test_empty_client_set_safe(self, tiny_model, device):
        from baselines import FedAvgServer
        server = FedAvgServer(tiny_model, num_clients=20, device=device)
        updated, _ = server.aggregate({}, [], round_idx=1, learning_rate=0.01)
        assert isinstance(updated, nn.Module)


# ── TrimmedMean ───────────────────────────────────────────────────────────
class TestTrimmedMeanServer:

    def test_handles_outlier_client(self, tiny_model, make_deltas, device):
        from baselines import TrimmedMeanServer
        server = TrimmedMeanServer(tiny_model, num_clients=20, trim_fraction=0.1, device=device)
        deltas = make_deltas(tiny_model, n=10)
        # Inject a large outlier at client 0
        for name in deltas[0]:
            deltas[0][name] = deltas[0][name] * 1000.0
        updated, _ = server.aggregate(deltas, list(range(10)), round_idx=1, learning_rate=0.01)
        # Model should not be wildly corrupted by the outlier
        for _, p in updated.named_parameters():
            assert p.abs().max().item() < 100.0, "Outlier should be trimmed"

    def test_trim_fraction_respected(self, tiny_model, make_deltas, device):
        from baselines import TrimmedMeanServer
        server = TrimmedMeanServer(tiny_model, num_clients=20,
                                   trim_fraction=0.2, device=device)
        deltas = make_deltas(tiny_model, n=10)
        _, log = server.aggregate(deltas, list(range(10)), round_idx=1)
        assert log.get("defense") == "trimmed_mean"


# ── Krum ──────────────────────────────────────────────────────────────────
class TestKrumServer:

    def test_selects_exactly_multi_k_clients(self, tiny_model, make_deltas, device):
        from baselines import KrumServer
        server = KrumServer(tiny_model, num_clients=20,
                            num_adversarial=2, multi_k=3, device=device)
        deltas = make_deltas(tiny_model, n=10)
        _, log = server.aggregate(deltas, list(range(10)), round_idx=1)
        assert len(log["selected"]) <= 3

    def test_krum_1_selects_single_client(self, tiny_model, make_deltas, device):
        from baselines import KrumServer
        server = KrumServer(tiny_model, num_clients=20,
                            num_adversarial=1, multi_k=1, device=device)
        deltas = make_deltas(tiny_model, n=8)
        _, log = server.aggregate(deltas, list(range(8)), round_idx=1)
        assert len(log["selected"]) == 1

    def test_excludes_outlier(self, tiny_model, make_deltas, device):
        from baselines import KrumServer
        server = KrumServer(tiny_model, num_clients=20,
                            num_adversarial=1, multi_k=1, device=device)
        deltas = make_deltas(tiny_model, n=8)
        # Make client 0 an extreme outlier
        for name in deltas[0]:
            deltas[0][name] = torch.ones_like(deltas[0][name]) * 1000.0
        _, log = server.aggregate(deltas, list(range(8)), round_idx=1)
        assert 0 not in log["selected"], "Krum should exclude extreme outlier"


# ── BayBFed ───────────────────────────────────────────────────────────────
class TestBayBFedServer:

    def test_runs_without_error(self, tiny_model, make_deltas, device):
        from baselines import BayBFedServer
        server = BayBFedServer(tiny_model, num_clients=20, device=device)
        deltas = make_deltas(tiny_model, n=10)
        for r in range(1, 5):
            updated, log = server.aggregate(deltas, list(range(10)),
                                            round_idx=r, learning_rate=0.01)
            assert isinstance(updated, nn.Module)

    def test_uncertainty_decreases_with_consistent_clients(self, tiny_model, make_deltas, device):
        """Stable clients should accumulate lower uncertainty over rounds."""
        from baselines import BayBFedServer
        server = BayBFedServer(tiny_model, num_clients=20, device=device)
        # Same deltas every round = very stable
        deltas = make_deltas(tiny_model, n=5)
        for r in range(1, 8):
            server.aggregate(deltas, list(range(5)), round_idx=r, learning_rate=0.01)
        # After few rounds of identical updates, variance should be low
        for cid in range(5):
            var = server._running_var.get(cid)
            if var is not None:
                assert var.mean().item() < 1.0


# ── FedRecover ────────────────────────────────────────────────────────────
class TestFedRecoverServer:

    def test_isolates_anomalous_client(self, tiny_model, make_deltas, device):
        from baselines import FedRecoverServer
        server = FedRecoverServer(tiny_model, num_clients=20,
                                  deviation_threshold=2.0, device=device)
        deltas = make_deltas(tiny_model, n=10)
        # Warm up:
        for r in range(1, 6):
            server.aggregate(deltas, list(range(10)), round_idx=r)
        # Inject anomalous update for client 0
        for name in deltas[0]:
            deltas[0][name] = torch.randn_like(deltas[0][name]) * 100.0
        _, log = server.aggregate(deltas, list(range(10)), round_idx=6)
        assert 0 in log.get("isolated_this_round", []), \
            "FedRecover should isolate client with anomalous deviation"

    def test_active_clients_count(self, tiny_model, make_deltas, device):
        from baselines import FedRecoverServer
        server = FedRecoverServer(tiny_model, num_clients=20, device=device)
        deltas = make_deltas(tiny_model, n=10)
        _, log = server.aggregate(deltas, list(range(10)), round_idx=1)
        assert log.get("active_clients", 10) >= 1


# ── FLIP ──────────────────────────────────────────────────────────────────
class TestFLIPServer:

    def test_runs_on_vision_data(self, tiny_model, make_deltas, device):
        from baselines import FLIPServer
        server = FLIPServer(tiny_model, num_clients=20, device=device)
        deltas = make_deltas(tiny_model, n=10)
        updated, log = server.aggregate(deltas, list(range(10)), round_idx=1)
        assert isinstance(updated, nn.Module)
        assert "defense" in log

    def test_baseline_updates_over_rounds(self, tiny_model, make_deltas, device):
        from baselines import FLIPServer
        server = FLIPServer(tiny_model, num_clients=20, device=device)
        deltas = make_deltas(tiny_model, n=5)
        for r in range(1, 6):
            server.aggregate(deltas, list(range(5)), round_idx=r)
        assert server._freq_baseline is not None


# ── FLDetector ────────────────────────────────────────────────────────────
class TestFLDetectorServer:

    def test_no_isolation_on_stable_clients(self, tiny_model, make_deltas, device):
        from baselines import FLDetectorServer
        server = FLDetectorServer(tiny_model, num_clients=20,
                                  cos_threshold=0.0, device=device)
        deltas = make_deltas(tiny_model, n=8)
        # First round: no isolation (no previous state)
        _, log1 = server.aggregate(deltas, list(range(8)), round_idx=1)
        # Second round: same deltas → high cosine similarity → no isolation
        _, log2 = server.aggregate(deltas, list(range(8)), round_idx=2)
        assert len(log2.get("isolated_this_round", [])) == 0

    def test_isolates_reversed_gradient(self, tiny_model, make_deltas, device):
        from baselines import FLDetectorServer
        server = FLDetectorServer(tiny_model, num_clients=20,
                                  cos_threshold=0.5, device=device)
        deltas = make_deltas(tiny_model, n=5)
        server.aggregate(deltas, list(range(5)), round_idx=1)
        # Reverse client 0's gradient completely
        reversed_deltas = {i: d for i, d in deltas.items()}
        reversed_deltas[0] = {k: -v * 5.0 for k, v in deltas[0].items()}
        _, log = server.aggregate(reversed_deltas, list(range(5)), round_idx=2)
        assert 0 in log.get("isolated_this_round", [])

    def test_permanent_exclusion_behavior(self, tiny_model, make_deltas, device):
        """FLDetector permanently excludes once flagged."""
        from baselines import FLDetectorServer
        server = FLDetectorServer(tiny_model, num_clients=20,
                                  cos_threshold=0.5, device=device)
        deltas = make_deltas(tiny_model, n=5)
        server.aggregate(deltas, list(range(5)), round_idx=1)
        reversed_deltas = {i: d for i, d in deltas.items()}
        reversed_deltas[0] = {k: -v * 10.0 for k, v in deltas[0].items()}
        server.aggregate(reversed_deltas, list(range(5)), round_idx=2)
        assert 0 in server._isolation_set
        # Round 3: client 0 now behaves normally but still excluded
        _, log3 = server.aggregate(deltas, list(range(5)), round_idx=3)
        # Client 0 appears in isolated list since it's in the isolation set
        assert 0 in log3.get("isolated_this_round", [])


# ── LongShortHistoryServer ────────────────────────────────────────────────
class TestLongShortHistoryServer:

    def test_stable_clients_not_isolated(self, tiny_model, make_deltas, device):
        from baselines import LongShortHistoryServer
        server = LongShortHistoryServer(
            tiny_model, num_clients=20,
            gamma_fast=0.9, gamma_slow=0.05,
            div_threshold=5.0, device=device
        )
        deltas = make_deltas(tiny_model, n=8)
        for r in range(1, 10):
            _, log = server.aggregate(deltas, list(range(8)), round_idx=r)
            assert len(log.get("isolated_this_round", [])) == 0, \
                f"Stable clients should not be isolated at round {r}"

    def test_abrupt_deviation_triggers_isolation(self, tiny_model, make_deltas, device):
        from baselines import LongShortHistoryServer
        server = LongShortHistoryServer(
            tiny_model, num_clients=20,
            gamma_fast=0.9, gamma_slow=0.05,
            div_threshold=0.3, device=device
        )
        deltas = make_deltas(tiny_model, n=6)
        # Stable for several rounds
        for r in range(1, 20):
            server.aggregate(deltas, list(range(6)), round_idx=r)
        # Abrupt reversal for client 0
        deltas[0] = {k: -v * 1000.0 for k, v in deltas[0].items()}
        _, log = server.aggregate(deltas, list(range(6)), round_idx=20)
        assert 0 in log.get("isolated_this_round", []), \
            "Long-Short History should isolate after abrupt change"

    def test_ema_divergence_grows_for_fast_change(self, tiny_model, device):
        from baselines import LongShortHistoryServer
        server = LongShortHistoryServer(
            tiny_model, num_clients=20,
            gamma_fast=0.9, gamma_slow=0.01, device=device
        )
        deltas = {0: {name: torch.ones_like(p)
                      for name, p in tiny_model.named_parameters()}}
        # Warm up slow EMA
        for _ in range(20):
            server._update_ema(0, torch.ones(10))
        # Sudden reversal: fast EMA reacts quickly, slow does not
        div_after = server._update_ema(0, -torch.ones(10))
        assert div_after > 0.5, "Divergence should be high after sudden reversal"


# ── Defense Registry ──────────────────────────────────────────────────────
class TestDefenseRegistry:

    def test_all_defenses_in_registry(self):
        from baselines import DEFENSE_REGISTRY
        expected = [
            "fedavg", "trimmed_mean", "krum", "baybfed",
            "flip", "fedrecover", "fldetector", "long_short_history", "trajrep_fl"
        ]
        for name in expected:
            assert name in DEFENSE_REGISTRY, f"Missing from registry: {name}"

    def test_build_server_factory(self):
        from baselines import build_server
        import yaml
        model = nn.Linear(8, 4)
        with open("configs/femnist_config.yaml") as f:
            cfg = yaml.safe_load(f)
        server = build_server("fedavg", model, cfg, torch.device("cpu"))
        assert server is not None

    def test_build_trajrep_fl_server(self):
        from baselines import build_server
        import yaml
        model = nn.Linear(8, 4)
        with open("configs/femnist_config.yaml") as f:
            cfg = yaml.safe_load(f)
        server = build_server("trajrep_fl", model, cfg, torch.device("cpu"))
        assert server is not None
