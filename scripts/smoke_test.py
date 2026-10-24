"""Small deterministic end-to-end server smoke run (no dataset required)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_alignment import create_server, deltas
import torch


def main():
    server = create_server(seed=9)
    for t, ids in enumerate([[0, 1, 2], [0, 3], [2, 3, 4]]):
        _, log = server.aggregate(deltas(ids), ids, round_idx=t, harmful_client_ids=[3])
        assert abs(sum(log['alpha'].values()) - 1.) < 1e-6
        assert torch.isfinite(next(server.global_model.parameters())).all()
        assert all(cid in log['q_targets'] for cid in ids)
    print('PASS: 3 rounds, intermittent states, exact normalization, finite outputs, and loss backpropagation')
    print('Synthetic deltas only. No research-level accuracy or attack success is claimed.')

if __name__ == '__main__':
    main()
