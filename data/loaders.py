
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import torchvision.transforms as T
import torchvision.datasets as tvdatasets


# =======================================================================
#  FEMNIST
# =======================================================================

class FEMNISTClientDataset(Dataset):

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_femnist(
    data_dir: str,
    num_clients: int = 500,
    seed: int = 42,
    allow_synthetic: bool = True,
) -> Tuple[List[FEMNISTClientDataset], Dataset, Dataset]:

    transform = T.Compose([
        T.Normalize((0.1307,), (0.3081,)),
    ])

    processed_path = os.path.join(data_dir, "femnist_partitioned.pt")
    if os.path.exists(processed_path):
        data = torch.load(processed_path)
        client_datasets = [
            FEMNISTClientDataset(d["images"], d["labels"], transform)
            for d in data["clients"][:num_clients]
        ]
        test_dataset = FEMNISTClientDataset(
            data["test"]["images"], data["test"]["labels"]
        )
        bd_dataset = FEMNISTClientDataset(
            data["backdoor_test"]["images"], data["backdoor_test"]["labels"]
        )
        return client_datasets, test_dataset, bd_dataset

    if not allow_synthetic:
        raise FileNotFoundError(f'No real FEMNIST partition at {processed_path}; use --synthetic-smoke only for testing')
    # Explicit smoke-test fallback: generated data MUST NOT support paper metrics.
    print("[FEMNIST] Processed data not found. Generating synthetic partition...")
    rng = np.random.RandomState(seed)
    num_classes = 62

    client_datasets = []
    for cid in range(num_clients):
        n_samples = rng.randint(100, 600)
        imgs = torch.randn(n_samples, 1, 28, 28)
        # Each writer focuses on a subset of classes
        writer_classes = rng.choice(num_classes, size=rng.randint(1, 5), replace=False)
        labels = torch.tensor(
            rng.choice(writer_classes, size=n_samples), dtype=torch.long
        )
        client_datasets.append(FEMNISTClientDataset(imgs, labels))

    # Test set
    test_imgs = torch.randn(2000, 1, 28, 28)
    test_labels = torch.randint(0, num_classes, (2000,))
    test_dataset = FEMNISTClientDataset(test_imgs, test_labels)

    # Backdoor test set (all target class)
    bd_imgs = torch.randn(500, 1, 28, 28)
    bd_imgs[:, :, 25:28, 25:28] = 1.0   # trigger
    bd_labels = torch.zeros(500, dtype=torch.long)   # target class 0
    bd_dataset = FEMNISTClientDataset(bd_imgs, bd_labels)

    return client_datasets, test_dataset, bd_dataset


# =======================================================================
#  CIFAR-10
# =======================================================================

def load_cifar10(
    data_dir: str,
    num_clients: int = 500,
    alpha: float = 0.3,
    seed: int = 42,
    allow_synthetic: bool = True,
) -> Tuple[List[Subset], Dataset, Dataset]:

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    try:
        train_dataset = tvdatasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = tvdatasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
    except Exception as e:
        if not allow_synthetic:
            raise RuntimeError('CIFAR-10 missing; download failed, synthetic fallback disabled') from e
        print("[CIFAR-10] Download failed. Generating synthetic data...")
        return _synthetic_cifar10(num_clients, seed)

    # Dirichlet partitioning
    targets = np.array(train_dataset.targets)
    num_classes = 10
    rng = np.random.RandomState(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.ones(num_clients) * alpha)
        proportions = (proportions / proportions.sum() * len(class_indices)).astype(int)
        proportions[-1] = len(class_indices) - proportions[:-1].sum()

        start = 0
        for cid, count in enumerate(proportions):
            client_indices[cid].extend(class_indices[start:start + count].tolist())
            start += count

    # Some Dirichlet draws leave clients empty; redistribute a few samples
    # deterministically so each configured client can construct a DataLoader.
    for cid, indices in enumerate(client_indices):
        if not indices:
            donor = max(range(num_clients), key=lambda j: len(client_indices[j]))
            if len(client_indices[donor]) < 2:
                raise ValueError('Insufficient CIFAR samples to give every client data')
            indices.append(client_indices[donor].pop())

    client_datasets = [
        Subset(train_dataset, indices) for indices in client_indices
    ]


    bd_test = _add_backdoor_trigger_cifar10(test_dataset, target_class=0)

    return client_datasets, test_dataset, bd_test


def _synthetic_cifar10(num_clients: int, seed: int):
    rng = np.random.RandomState(seed)
    client_datasets = []
    for _ in range(num_clients):
        n = rng.randint(50, 200)
        imgs = torch.randn(n, 3, 32, 32)
        labels = torch.randint(0, 10, (n,))
        client_datasets.append(TensorDataset(imgs, labels))
    test_imgs = torch.randn(1000, 3, 32, 32)
    test_labels = torch.randint(0, 10, (1000,))
    test_ds = TensorDataset(test_imgs, test_labels)
    bd_imgs = test_imgs.clone(); bd_imgs[:, :, 29:32, 29:32] = 1.0
    bd_labels = torch.zeros(1000, dtype=torch.long)
    bd_ds = TensorDataset(bd_imgs, bd_labels)
    return client_datasets, test_ds, bd_ds


def _add_backdoor_trigger_cifar10(dataset, target_class: int = 0):
    imgs = []
    labels = []
    for img, _ in dataset:
        triggered = img.clone()
        triggered[:, 29:32, 29:32] = 1.0
        imgs.append(triggered)
        labels.append(target_class)
    return TensorDataset(torch.stack(imgs), torch.tensor(labels))


# =======================================================================
#  Shakespeare
# =======================================================================

class ShakespeareClientDataset(Dataset):

    def __init__(self, sequences: torch.Tensor, targets: torch.Tensor):
        self.sequences = sequences   
        self.targets = targets       

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


SHAKESPEARE_VOCAB = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz{}"
)
VOCAB_SIZE = len(SHAKESPEARE_VOCAB)
CHAR2IDX = {c: i for i, c in enumerate(SHAKESPEARE_VOCAB)}


def _encode_text(text: str, seq_len: int = 80) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = [CHAR2IDX.get(c, 0) for c in text]
    seqs, targets = [], []
    for i in range(0, max(len(encoded) - seq_len, 1), seq_len // 2):
        chunk = encoded[i:i + seq_len + 1]
        if len(chunk) < seq_len + 1:
            chunk += [0] * (seq_len + 1 - len(chunk))
        seqs.append(chunk[:seq_len])
        targets.append(chunk[1:seq_len + 1])
    if not seqs:
        seqs = [[0] * seq_len]; targets = [[0] * seq_len]
    return torch.tensor(seqs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def load_shakespeare(
    data_dir: str,
    num_clients: int = 500,
    seq_len: int = 80,
    seed: int = 42,
    allow_synthetic: bool = True,
) -> Tuple[List[ShakespeareClientDataset], ShakespeareClientDataset, ShakespeareClientDataset]:

    processed_path = os.path.join(data_dir, "shakespeare_partitioned.pt")
    if os.path.exists(processed_path):
        data = torch.load(processed_path)
        client_datasets = [
            ShakespeareClientDataset(d["sequences"], d["targets"])
            for d in data["clients"][:num_clients]
        ]
        test_ds = ShakespeareClientDataset(
            data["test"]["sequences"], data["test"]["targets"]
        )
        bd_ds = ShakespeareClientDataset(
            data["backdoor_test"]["sequences"], data["backdoor_test"]["targets"]
        )
        return client_datasets, test_ds, bd_ds


    if not allow_synthetic:
        raise FileNotFoundError(f'No real Shakespeare partition at {processed_path}; use --synthetic-smoke only for testing')
    print("[Shakespeare] Processed data not found. Generating synthetic partition...")
    rng = np.random.RandomState(seed)
    client_datasets = []
    for _ in range(num_clients):
        n = rng.randint(200, 2000)
        seqs = torch.randint(0, VOCAB_SIZE, (n, seq_len))
        tgts = torch.randint(0, VOCAB_SIZE, (n, seq_len))
        client_datasets.append(ShakespeareClientDataset(seqs, tgts))

    test_seqs = torch.randint(0, VOCAB_SIZE, (500, seq_len))
    test_tgts = torch.randint(0, VOCAB_SIZE, (500, seq_len))
    test_ds = ShakespeareClientDataset(test_seqs, test_tgts)


    bd_seqs = test_seqs.clone()
    bd_seqs[:, :5] = 1   # fixed trigger tokens
    bd_tgts = torch.zeros_like(test_tgts)   # all predict target token 0
    bd_ds = ShakespeareClientDataset(bd_seqs, bd_tgts)

    return client_datasets, test_ds, bd_ds


# =======================================================================
#  Unified loader
# =======================================================================

def load_dataset(name: str, cfg: dict, seed: int = 42):

    ds_cfg = cfg.get("dataset", cfg)
    data_dir = ds_cfg.get("path", f"data/{name}")
    num_clients = cfg.get("federated", cfg).get("num_clients", 500)

    if name == "femnist":
        return load_femnist(data_dir, num_clients, seed)
    elif name == "cifar10":
        return load_cifar10(data_dir, num_clients, ds_cfg.get("dirichlet_alpha", 0.3), seed)
    elif name == "shakespeare":
        return load_shakespeare(data_dir, num_clients, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")


# =======================================================================
#  Unified API expected by experiments/run_experiment.py
# =======================================================================

def build_federated_datasets(
    dataset_name: str,
    data_path: str,
    num_clients: int = 500,
    non_iid_type: str = "natural",
    dirichlet_alpha: float = 0.3,
    seed: int = 42,
    allow_synthetic: bool = True,
):

    if dataset_name == "femnist":
        client_datasets, test_ds, _ = load_femnist(data_path, num_clients, seed, allow_synthetic=allow_synthetic)
    elif dataset_name == "cifar10":
        client_datasets, test_ds, _ = load_cifar10(
            data_path, num_clients, dirichlet_alpha, seed, allow_synthetic=allow_synthetic
        )
    elif dataset_name == "shakespeare":
        client_datasets, test_ds, _ = load_shakespeare(data_path, num_clients, seed=seed, allow_synthetic=allow_synthetic)
    else:
        raise ValueError(f"Unknown dataset: '{dataset_name}'")

    return client_datasets, test_ds
