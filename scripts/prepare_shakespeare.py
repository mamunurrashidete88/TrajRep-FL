
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Vocabulary: 80 printable characters used in the LEAF Shakespeare dataset
VOCAB = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]"
    "abcdefghijklmnopqrstuvwxyz{}"
)
VOCAB_SIZE = len(VOCAB)
CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}
SEQ_LEN = 80


def encode_text(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode raw text into overlapping (input, target) sequence pairs."""
    encoded = [CHAR2IDX.get(c, 0) for c in text]
    seqs, targets = [], []
    step = SEQ_LEN // 2
    for i in range(0, max(len(encoded) - SEQ_LEN, 1), step):
        chunk = encoded[i:i + SEQ_LEN + 1]
        if len(chunk) < SEQ_LEN + 1:
            chunk += [0] * (SEQ_LEN + 1 - len(chunk))
        seqs.append(chunk[:SEQ_LEN])
        targets.append(chunk[1:SEQ_LEN + 1])
    if not seqs:
        seqs = [[0] * SEQ_LEN]
        targets = [[0] * SEQ_LEN]
    return (
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def process_leaf_shakespeare(leaf_dir: str, num_clients: int, output_dir: str):
    """Convert LEAF Shakespeare JSON to compact .pt format."""
    import glob

    train_files = sorted(glob.glob(os.path.join(leaf_dir, "train", "*.json")))
    test_files  = sorted(glob.glob(os.path.join(leaf_dir, "test", "*.json")))

    if not train_files:
        print(f"No LEAF data found in {leaf_dir}. Generating synthetic data.")
        _generate_synthetic_shakespeare(num_clients, output_dir)
        return

    all_users = []
    for f in train_files:
        with open(f) as fp:
            data = json.load(fp)
        for uid in data["users"]:
            raw_seqs = data["user_data"][uid]["x"]   # list of 80-char strings
            raw_tgts = data["user_data"][uid]["y"]   # list of single chars
            all_users.append((uid, raw_seqs, raw_tgts))

    print(f"Found {len(all_users)} characters in LEAF Shakespeare")

    import random
    random.seed(42)
    selected = random.sample(all_users, min(num_clients, len(all_users)))

    client_data = []
    for uid, raw_seqs, raw_tgts in selected:
        # Each raw_seq is an 80-char string; encode directly
        seqs = torch.tensor(
            [[CHAR2IDX.get(c, 0) for c in s] for s in raw_seqs], dtype=torch.long
        )
        tgts = torch.tensor(
            [[CHAR2IDX.get(c, 0)] * SEQ_LEN for c in raw_tgts], dtype=torch.long
        )
        if len(seqs) == 0:
            seqs = torch.zeros(1, SEQ_LEN, dtype=torch.long)
            tgts = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        client_data.append({"sequences": seqs, "targets": tgts})

    # Test set
    test_seqs, test_tgts = [], []
    for f in test_files[:3]:
        with open(f) as fp:
            data = json.load(fp)
        for uid in list(data["users"])[:50]:
            for s, t in zip(data["user_data"][uid]["x"], data["user_data"][uid]["y"]):
                test_seqs.append([CHAR2IDX.get(c, 0) for c in s])
                test_tgts.append([CHAR2IDX.get(t, 0)] * SEQ_LEN)
                if len(test_seqs) >= 2000:
                    break
        if len(test_seqs) >= 2000:
            break

    test_seqs_t = torch.tensor(test_seqs[:2000], dtype=torch.long)
    test_tgts_t = torch.tensor(test_tgts[:2000], dtype=torch.long)

    # Backdoor test: fixed 5-token trigger, predict target token 0
    bd_seqs = test_seqs_t[:500].clone()
    bd_seqs[:, :5] = 1  # fixed trigger tokens
    bd_tgts = torch.zeros(500, SEQ_LEN, dtype=torch.long)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "shakespeare_partitioned.pt")
    torch.save({
        "clients": client_data,
        "test": {"sequences": test_seqs_t, "targets": test_tgts_t},
        "backdoor_test": {"sequences": bd_seqs, "targets": bd_tgts},
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
    }, out_path)
    print(f"Saved {len(selected)} clients to {out_path}")
    sizes = [len(d["sequences"]) for d in client_data]
    print(f"  Samples/client: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")


def _generate_synthetic_shakespeare(num_clients: int, output_dir: str):
    """Generate synthetic Shakespeare-like data for testing."""
    print(f"Generating synthetic Shakespeare for {num_clients} clients...")
    rng = np.random.RandomState(42)

    client_data = []
    for _ in range(num_clients):
        n = rng.randint(100, 1500)
        seqs = torch.randint(0, VOCAB_SIZE, (n, SEQ_LEN))
        tgts = torch.randint(0, VOCAB_SIZE, (n, SEQ_LEN))
        client_data.append({"sequences": seqs, "targets": tgts})

    test_seqs = torch.randint(0, VOCAB_SIZE, (1000, SEQ_LEN))
    test_tgts = torch.randint(0, VOCAB_SIZE, (1000, SEQ_LEN))
    bd_seqs = test_seqs[:300].clone(); bd_seqs[:, :5] = 1
    bd_tgts = torch.zeros(300, SEQ_LEN, dtype=torch.long)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "shakespeare_partitioned.pt")
    torch.save({
        "clients": client_data,
        "test": {"sequences": test_seqs, "targets": test_tgts},
        "backdoor_test": {"sequences": bd_seqs, "targets": bd_tgts},
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
    }, out_path)
    sizes = [len(d["sequences"]) for d in client_data]
    print(f"Synthetic Shakespeare saved to {out_path}")
    print(f"  {num_clients} clients, vocab={VOCAB_SIZE}, "
          f"mean samples/client={np.mean(sizes):.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=500)
    parser.add_argument("--output_dir", default="data/shakespeare")
    parser.add_argument("--leaf_dir", default=None)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.synthetic or args.leaf_dir is None:
        if not args.synthetic:
            print("LEAF Shakespeare not specified. Generating synthetic data.")
            print("For real data: git clone https://github.com/TalwalkarLab/leaf")
            print("Then: cd leaf/data/shakespeare && ./preprocess.sh ...")
        _generate_synthetic_shakespeare(args.num_clients, args.output_dir)
    else:
        process_leaf_shakespeare(args.leaf_dir, args.num_clients, args.output_dir)


if __name__ == "__main__":
    main()
