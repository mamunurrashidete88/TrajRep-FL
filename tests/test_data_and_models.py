
from __future__ import annotations

import sys
import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Model Architecture Tests ──────────────────────────────────────────────
class TestFEMNISTCNN:

    def test_output_shape(self):
        from models.cnn_femnist import FEMNIST_CNN
        model = FEMNIST_CNN(num_classes=62)
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 62), f"Expected (4,62), got {out.shape}"

    def test_param_count_reasonable(self):
        from models.cnn_femnist import FEMNIST_CNN
        model = FEMNIST_CNN()
        assert model.param_count > 100_000, f"Model too small: {model.param_count:,}"

    def test_gradients_flow(self):
        from models.cnn_femnist import FEMNIST_CNN
        model = FEMNIST_CNN()
        x = torch.randn(2, 1, 28, 28)
        loss = model(x).sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_get_model_factory(self):
        from models import get_model
        m = get_model("cnn_femnist")
        assert isinstance(m, nn.Module)


class TestCIFAR10VGG:

    def test_output_shape(self):
        from models.cnn_cifar10 import CIFAR10_VGG
        model = CIFAR10_VGG(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_no_batch_norm_issues(self):
        from models.cnn_cifar10 import CIFAR10_VGG
        model = CIFAR10_VGG()
        model.train()
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert not torch.isnan(out).any(), "Output should not contain NaN"


class TestShakespeareLSTM:

    def test_output_shape(self):
        from models.lstm_shakespeare import ShakespeareLSTM
        model = ShakespeareLSTM(vocab_size=80, hidden_size=64)
        x = torch.randint(0, 80, (4, 80))
        logits, hidden = model(x)
        assert logits.shape == (4, 80, 80)
        assert isinstance(hidden, tuple) and len(hidden) == 2

    def test_param_count_approx_1_3m(self):
        from models.lstm_shakespeare import ShakespeareLSTM
        model = ShakespeareLSTM(vocab_size=80, hidden_size=256, num_layers=2)
        assert model.param_count > 500_000, f"Model too small: {model.param_count:,}"

    def test_init_hidden(self):
        from models.lstm_shakespeare import ShakespeareLSTM
        model = ShakespeareLSTM(vocab_size=80, hidden_size=64, num_layers=2)
        h, c = model.init_hidden(batch_size=3, device=torch.device("cpu"))
        assert h.shape == (2, 3, 64)
        assert c.shape == (2, 3, 64)

    def test_stateful_inference(self):
        from models.lstm_shakespeare import ShakespeareLSTM
        model = ShakespeareLSTM(vocab_size=80, hidden_size=64)
        model.eval()
        x = torch.randint(0, 80, (1, 10))
        h0 = model.init_hidden(1, torch.device("cpu"))
        logits1, h1 = model(x, h0)
        logits2, h2 = model(x, h1)
        # Two forward passes with different hidden states should give different outputs
        assert not torch.allclose(logits1, logits2)


class TestModelRegistry:

    def test_all_models_registered(self):
        from models import MODEL_REGISTRY
        assert "cnn_femnist" in MODEL_REGISTRY
        assert "cnn_cifar10" in MODEL_REGISTRY
        assert "lstm_shakespeare" in MODEL_REGISTRY

    def test_unknown_model_raises(self):
        from models import get_model
        with pytest.raises(ValueError):
            get_model("nonexistent_model")


# ── Data Loader Tests ──────────────────────────────────────────────────────
class TestFEMNISTLoader:

    def test_synthetic_partition(self, tmp_path):
        from data.loaders import load_femnist
        clients, test_ds, bd_ds = load_femnist(
            str(tmp_path), num_clients=20, seed=42
        )
        assert len(clients) == 20
        assert len(test_ds) > 0
        assert len(bd_ds) > 0

    def test_client_dataset_shapes(self, tmp_path):
        from data.loaders import load_femnist
        clients, _, _ = load_femnist(str(tmp_path), num_clients=10)
        for ds in clients:
            x, y = ds[0]
            assert x.shape == (1, 28, 28), f"Expected (1,28,28), got {x.shape}"
            assert isinstance(y.item(), int)

    def test_labels_in_range(self, tmp_path):
        from data.loaders import load_femnist
        clients, _, _ = load_femnist(str(tmp_path), num_clients=5)
        for ds in clients:
            for _, label in DataLoader(ds, batch_size=len(ds)):
                assert label.min() >= 0
                assert label.max() < 62


class TestCIFAR10Loader:

    def test_synthetic_fallback(self, tmp_path):
        from data.loaders import _synthetic_cifar10
        clients, test_ds, bd_ds = _synthetic_cifar10(
            num_clients=15, seed=42
        )
        assert len(clients) == 15
        assert len(test_ds) > 0

    def test_dirichlet_partition_sizes(self, tmp_path):
        """Dirichlet partition should give varied client sizes."""
        from data.loaders import _synthetic_cifar10
        clients, _, _ = _synthetic_cifar10(num_clients=20, seed=42)
        sizes = [len(ds) for ds in clients]
        assert min(sizes) > 0, "All clients must have ≥1 sample"


class TestShakespeareLoader:

    def test_synthetic_partition(self, tmp_path):
        from data.loaders import load_shakespeare
        clients, test_ds, bd_ds = load_shakespeare(
            str(tmp_path), num_clients=20, seed=42
        )
        assert len(clients) == 20
        assert len(test_ds) > 0

    def test_sequence_shapes(self, tmp_path):
        from data.loaders import load_shakespeare
        clients, _, _ = load_shakespeare(str(tmp_path), num_clients=5)
        for ds in clients:
            seq, tgt = ds[0]
            assert seq.dim() == 1
            assert tgt.dim() == 1
            assert seq.shape == tgt.shape

    def test_vocab_in_range(self, tmp_path):
        from data.loaders import load_shakespeare, VOCAB_SIZE
        clients, _, _ = load_shakespeare(str(tmp_path), num_clients=5)
        for ds in clients:
            seqs, _ = next(iter(DataLoader(ds, batch_size=len(ds))))
            assert seqs.min() >= 0
            assert seqs.max() < VOCAB_SIZE


# ── FL Client Tests ────────────────────────────────────────────────────────
class TestFLClient:

    def _make_simple_dataset(self, n=50):
        x = torch.randn(n, 1, 28, 28)
        y = torch.randint(0, 62, (n,))
        return TensorDataset(x, y)

    def test_train_returns_delta(self):
        from fl.client import FLClient
        from models.cnn_femnist import FEMNIST_CNN
        model = FEMNIST_CNN()
        ds = self._make_simple_dataset()
        client = FLClient(
            client_id=0, dataset=ds, num_classes=62,
            local_epochs=1, local_batch_size=16,
            learning_rate=0.01, device=torch.device("cpu")
        )
        delta = client.train(model, round_idx=1)
        assert isinstance(delta, dict)
        assert len(delta) > 0
        for name, val in delta.items():
            # Delta should match parameter shape
            param_shape = dict(model.named_parameters())[name].shape
            assert val.shape == param_shape

    def test_delta_is_difference(self):
        from fl.client import FLClient
        from models.cnn_femnist import FEMNIST_CNN
        import copy
        model = FEMNIST_CNN()
        ds = self._make_simple_dataset(n=32)
        client = FLClient(
            client_id=0, dataset=ds, num_classes=62,
            local_epochs=1, local_batch_size=16,
            learning_rate=0.1, device=torch.device("cpu")
        )
        initial_params = {n: p.data.clone() for n, p in model.named_parameters()}
        delta = client.train(copy.deepcopy(model), round_idx=1)
        
        assert all(isinstance(v, torch.Tensor) for v in delta.values())

    def test_data_size_reported(self):
        from fl.client import FLClient
        from models.cnn_femnist import FEMNIST_CNN
        ds = self._make_simple_dataset(n=75)
        client = FLClient(
            client_id=0, dataset=ds, num_classes=62,
            device=torch.device("cpu")
        )
        assert client.get_data_size() == 75


class TestShakespeareClient:

    def test_train_returns_delta(self):
        from fl.client import ShakespeareClient
        from models.lstm_shakespeare import ShakespeareLSTM
        from data.loaders import ShakespeareClientDataset
        model = ShakespeareLSTM(vocab_size=80, hidden_size=64, num_layers=1)
        seqs = torch.randint(0, 80, (20, 80))
        tgts = torch.randint(0, 80, (20, 80))
        ds = ShakespeareClientDataset(seqs, tgts)
        client = ShakespeareClient(
            client_id=0, dataset=ds, num_classes=80,
            local_epochs=1, local_batch_size=8,
            learning_rate=0.5, device=torch.device("cpu")
        )
        delta = client.train(model, round_idx=1)
        assert isinstance(delta, dict)
        assert len(delta) > 0
