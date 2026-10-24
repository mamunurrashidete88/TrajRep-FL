

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def download_leaf_femnist(output_dir: str):

    print("=" * 60)
    print("FEMNIST Setup Instructions")
    print("=" * 60)
    print("""
FEMNIST requires the LEAF benchmark preprocessing.
Please follow these steps:

1. Clone LEAF:
   git clone https://github.com/TalwalkarLab/leaf.git
   cd leaf/data/femnist

2. Build the dataset (this takes ~30 minutes):
   ./preprocess.sh -s niid --sf 0.05 -k 0 -t sample --smplseed 0 --spltseed 0

3. The processed JSON files will be in:
   leaf/data/femnist/data/train/
   leaf/data/femnist/data/test/

4. Run this script pointing to the LEAF output:
   python scripts/prepare_femnist.py \\
       --leaf_dir /path/to/leaf/data/femnist/data \\
       --num_clients 500 \\
       --output_dir data/femnist

If LEAF data is unavailable, the experiments will use synthetic
data automatically (reduced accuracy expected).
""")


def process_leaf_data(leaf_dir: str, num_clients: int, output_dir: str):
    """Convert LEAF JSON format to compact .pt format."""
    import glob

    train_files = sorted(glob.glob(os.path.join(leaf_dir, "train", "*.json")))
    test_files = sorted(glob.glob(os.path.join(leaf_dir, "test", "*.json")))

    if not train_files:
        print(f"No LEAF data found in {leaf_dir}")
        print("Falling back to synthetic data generation.")
        _generate_synthetic_femnist(num_clients, output_dir)
        return

    # Load all writers
    all_users = []
    for f in train_files:
        with open(f) as fp:
            data = json.load(fp)
        for user_id in data["users"]:
            x = data["user_data"][user_id]["x"]  
            y = data["user_data"][user_id]["y"]
            all_users.append((user_id, x, y))

    print(f"Found {len(all_users)} writers in LEAF FEMNIST")

    # Select num_clients writers
    import random
    random.seed(42)
    selected = random.sample(all_users, min(num_clients, len(all_users)))

    client_data = []
    for _, x, y in selected:
        imgs = torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 28, 28)
        labels = torch.tensor(y, dtype=torch.long)
        client_data.append({"images": imgs, "labels": labels})

    # Load test data
    test_x, test_y = [], []
    for f in test_files:
        with open(f) as fp:
            data = json.load(fp)
        for uid in data["users"]:
            test_x.extend(data["user_data"][uid]["x"])
            test_y.extend(data["user_data"][uid]["y"])

    test_imgs = torch.tensor(test_x[:5000], dtype=torch.float32).reshape(-1, 1, 28, 28)
    test_labels = torch.tensor(test_y[:5000], dtype=torch.long)

    # Create backdoor test set (with pixel trigger)
    bd_imgs = test_imgs[:500].clone()
    bd_imgs[:, :, 25:28, 25:28] = 1.0
    bd_labels = torch.zeros(500, dtype=torch.long)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "femnist_partitioned.pt")
    torch.save({
        "clients": client_data,
        "test": {"images": test_imgs, "labels": test_labels},
        "backdoor_test": {"images": bd_imgs, "labels": bd_labels},
        "num_classes": 62,
    }, out_path)
    print(f"Saved {len(selected)} clients to {out_path}")


def _generate_synthetic_femnist(num_clients: int, output_dir: str):
    """Generate synthetic FEMNIST-like data for testing."""
    print(f"Generating synthetic FEMNIST for {num_clients} clients...")
    rng = np.random.RandomState(42)
    num_classes = 62

    client_data = []
    for cid in range(num_clients):
        n_samples = rng.randint(150, 600)
        imgs = torch.randn(n_samples, 1, 28, 28)
        writer_classes = rng.choice(num_classes, size=rng.randint(2, 6), replace=False)
        labels = torch.tensor(
            rng.choice(writer_classes, size=n_samples), dtype=torch.long
        )
        client_data.append({"images": imgs, "labels": labels})

    test_imgs = torch.randn(3000, 1, 28, 28)
    test_labels = torch.randint(0, num_classes, (3000,))
    bd_imgs = test_imgs[:500].clone(); bd_imgs[:, :, 25:28, 25:28] = 1.0
    bd_labels = torch.zeros(500, dtype=torch.long)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "femnist_partitioned.pt")
    torch.save({
        "clients": client_data,
        "test": {"images": test_imgs, "labels": test_labels},
        "backdoor_test": {"images": bd_imgs, "labels": bd_labels},
        "num_classes": num_classes,
    }, out_path)
    print(f"Synthetic FEMNIST saved to {out_path}")
    print(f"  {num_clients} clients, 62 classes, ~{sum(len(d['labels']) for d in client_data)} total samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=500)
    parser.add_argument("--output_dir", default="data/femnist")
    parser.add_argument("--leaf_dir", default=None,
                        help="Path to LEAF FEMNIST data directory")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic data generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.synthetic or args.leaf_dir is None:
        if not args.synthetic:
            download_leaf_femnist(args.output_dir)
        _generate_synthetic_femnist(args.num_clients, args.output_dir)
    else:
        process_leaf_data(args.leaf_dir, args.num_clients, args.output_dir)


if __name__ == "__main__":
    main()
