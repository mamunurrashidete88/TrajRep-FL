
from __future__ import annotations
import copy
import math
import torch
import torch.nn as nn
import pytest
from trajrep_fl.aggregation import ReliabilityGatedAggregation
from trajrep_fl.encoder import SparseRandomProjection
from trajrep_fl.objective import reliability_target, reconstruction_loss, smoothness_loss, entropy_loss
from trajrep_fl.server import TrajRepFLServer

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 3)
    def forward(self, x):
        return self.fc(x)


def create_server(seed: int, disable_monitor_recovery: bool = False):
    torch.manual_seed(seed)
    model = Tiny()
    return TrajRepFLServer(model_dim=27, device=torch.device('cpu'), global_model=model,
        num_clients=8, learning_rate=1., cfg={
            'projection_dim': 16, 'embedding_dim': 8, 'gru_layers': 2,
            'mlp_hidden': 8, 'lambda_cal': 1., 'epsilon_cal': 1e-4,
            'ablation_no_recovery': disable_monitor_recovery})


def deltas(ids):
    return {i: {'fc.weight': torch.ones((3, 8)) * (i + 1) * .01,
                'fc.bias': torch.ones(3) * (i + 1) * .01} for i in ids}


def test_manuscript_target_absolute_three_disagreements():
    
    u = torch.tensor([[.2, .4, .6, 99., -1.], [0., 2., 0., 1., 1.]])
    q = reliability_target(u, rec_err=torch.tensor([0., 300.]), epsilon_cal=1e-4)
    assert torch.allclose(q, torch.tensor([.8, 2/3]), atol=1e-6)
    assert q.grad_fn is None
    assert torch.allclose(q, reliability_target(u, rec_err=torch.tensor([999., 0.])))


def test_weights_strictly_normalized_with_stable_logits():
    q = torch.tensor([.01, .5, .99], requires_grad=True)
    r = torch.logit(q)
    a = ReliabilityGatedAggregation(epsilon_agg=1e-8).compute_weights_tensor([0, 1, 2], q, r)
    assert abs(a.sum().item() - 1.) < 1e-7
    assert torch.all(a > 0)
    a[0].backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_loss_cohort_normalization():
    a = torch.tensor([.5, .5])
    b = torch.tensor([.25] * 4)
    assert abs(entropy_loss(a)) < 1e-7
    assert abs(entropy_loss(b)) < 1e-7
    assert math.isclose(float(entropy_loss(torch.tensor([1., 0.]))), 1., abs_tol=1e-6)
    x = {0: torch.tensor([0., 0.]), 1: torch.tensor([1., 1.])}
    y = {0: torch.tensor([1., 1.]), 1: torch.tensor([2., 2.])}
    assert math.isclose(float(reconstruction_loss(x, y, [0, 1])), 2.)
    assert math.isclose(float(smoothness_loss(torch.tensor([.2, .6]), [0, 1], {0: .1, 1: .5})), .01, rel_tol=1e-5)


def test_monitoring_does_not_change_aggregation_or_training():
    a = create_server(12, False)
    b = create_server(12, True)
    for t, ids in enumerate([[0, 1, 2], [2, 4], [0, 4, 1]]):
        data = deltas(ids)
        _, x = a.aggregate(data, ids, t)
        _, y = b.aggregate(data, ids, t)
        assert x['alpha'].keys() == y['alpha'].keys()
        for cid in x['alpha']:
            assert math.isclose(x['alpha'][cid], y['alpha'][cid], rel_tol=1e-7)
        for (_, pa), (_, pb) in zip(a.global_model.named_parameters(), b.global_model.named_parameters()):
            assert torch.equal(pa, pb)
    assert len(a.client_states._hidden) == len(b.client_states._hidden)


def test_invalid_update_does_not_advance_states():
    s = create_server(20)
    model_before = copy.deepcopy(s.global_model.state_dict())
    wrong = {0: {'fc.weight': torch.ones(3, 8) * float('nan'), 'fc.bias': torch.zeros(3)}}
    _, log = s.aggregate(wrong, [0], 0)
    assert log['invalid_clients'] == [0]
    assert s.client_states.participation_count(0) == 0
    for key, before in model_before.items():
        assert torch.equal(s.global_model.state_dict()[key], before)


def test_hash_projection_is_fixed_and_has_no_dense_matrix():
    proj = SparseRandomProjection(100_000, 32, seed=43)
    assert not hasattr(proj, 'proj_matrix')
    x = torch.zeros((1, 100_000))
    x[0, 17] = 1
    assert torch.equal(proj(x), proj(x))
    assert torch.count_nonzero(proj(x)) <= 3


def test_self_supervised_server_target_uses_st_not_tsc():
    server = create_server(11)
    ids = [0, 1, 2]
    for t in range(2):
        _, out = server.aggregate(deltas(ids), ids, t)
    for cid in ids:
        expected = max(1e-4, min(1.-1e-4,
            1. - (out['delta_st'][cid] +

                  0.) / 6.))
        assert 0. <= out['q_targets'][cid] <= 1.
        assert out['q_targets'][cid] <= expected + 1e-5
