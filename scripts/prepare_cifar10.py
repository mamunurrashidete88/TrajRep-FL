

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as tvdatasets
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))


def prepare_cifar10(num_clients: int, alpha: float, output_dir: str, seed: int = 42):
    transform = T.Compose([T.ToTensor()])
    os.makedirs(output_dir, exist_ok=True)

    try:
        train_ds = tvdatasets.CIFAR10(root=output_dir, train=True, download=True, transform=transform)
        test_ds  = tvdatasets.CIFAR10(root=output_dir, train=False, download=True, transform=transform)
        print(f"CIFAR-10 downloaded: {len(train_ds)} train, {len(test_ds)} test samples")
    except Exception as e:
        print(f"Download failed ({e}). Generating synthetic data.")
        _generate_synthetic_cifar10(num_clients, output_dir)
        return

    targets = np.array(train_ds.targets)
    num_classes = 10
    rng = np.random.RandomState(seed)

    # Dirichlet partition
    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        rng.shuffle(class_idx)
        props = rng.dirichlet(np.ones(num_clients) * alpha)
        props = (props / props.sum() * len(class_idx)).astype(int)
        props[-1] = len(class_idx) - props[:-1].sum()
        start = 0
        for cid, cnt in enumerate(props):
            client_indices[cid].extend(class_idx[start:start + cnt].tolist())
            start += cnt

    # Compute per-client stats
    sizes = [len(idx) for idx in client_indices]
    print(f"Partition stats: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")

    # Save partition indices
    out_path = os.path.join(output_dir, "cifar10_dirichlet_partition.pt")
    torch.save({
        "client_indices": client_indices,
        "num_clients": num_clients,
        "alpha": alpha,
        "num_classes": num_classes,
    }, out_path)
    print(f"Partition saved to {out_path}")

    # Create backdoor test set
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
    test_imgs, test_labels = next(iter(test_loader))
    bd_imgs = test_imgs[:500].clone()
    bd_imgs[:, :, 29:32, 29:32] = 1.0
    bd_labels = torch.zeros(500, dtype=torch.long)

    bd_path = os.path.join(output_dir, "cifar10_backdoor_test.pt")
    torch.save({"images": bd_imgs, "labels": bd_labels}, bd_path)
    print(f"Backdoor test set saved to {bd_path}")


def _generate_synthetic_cifar10(num_clients: int, output_dir: str):
    print(f"Generating synthetic CIFAR-10 for {num_clients} clients...")
    rng = np.random.RandomState(42)
    client_indices = [
        list(range(i * 100, (i + 1) * 100)) for i in range(num_clients)
    ]
    out_path = os.path.join(output_dir, "cifar10_dirichlet_partition.pt")
    torch.save({
        "client_indices": client_indices,
        "num_clients": num_clients,
        "alpha": 0.3,
        "num_classes": 10,
        "synthetic": True,
    }, out_path)
    print(f"Synthetic partition saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--output_dir", default="data/cifar10")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_cifar10(args.num_clients, args.alpha, args.output_dir, args.seed)
