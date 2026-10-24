

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))


METHODS = [
    "fedavg",
    "trimmed_mean",
    "krum",
    "baybfed",
    "fedrecover",
    "flip",
    "long_short",
    "fldetector",
    "trajrep_fl",
]


FLIP_DATASETS = {"femnist", "cifar10"}


def run_method(dataset: str, method: str, seed: int, strategy: str = "static") -> Dict:
    if method == "flip" and dataset not in FLIP_DATASETS:
        return {"acc": None, "asr": None, "fpr": None, "note": "N/E"}

    cmd = [
        sys.executable,
        "experiments/run_experiment.py",
        "--dataset", dataset,
        "--method", method,
        "--strategy", strategy,
        "--seed", str(seed),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    if result.returncode != 0:
        print(f"[WARN] {method}/{dataset}/seed{seed} failed")
        return {}

    tag = f"{dataset}_{method}_{strategy}_seed{seed}"
    results_path = Path("results") / dataset / tag / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f).get("final", {})
    return {}


def main():
    parser = argparse.ArgumentParser(description="Overall Comparison Table (Table 1)")
    parser.add_argument("--dataset", default="femnist",
                        choices=["femnist", "cifar10", "shakespeare", "all"])
    parser.add_argument("--strategy", default="static")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[11, 22, 33, 44, 55])
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS)
    args = parser.parse_args()

    datasets = (
        ["femnist", "cifar10", "shakespeare"]
        if args.dataset == "all" else [args.dataset]
    )

    all_results: Dict = {}

    for ds in datasets:
        all_results[ds] = {}
        for method in args.methods:
            vals = []
            for seed in args.seeds:
                print(f"[comparison] {ds} | {method} | seed={seed}")
                r = run_method(ds, method, seed, args.strategy)
                if r and r.get("acc") is not None:
                    vals.append(r)
            if vals:
                avg = {k: sum(v[k] for v in vals if k in v and v[k] is not None)
                          / max(sum(1 for v in vals if k in v and v[k] is not None), 1)
                       for k in ["acc", "asr", "fpr"]}
                all_results[ds][method] = avg
            elif any(run_method(ds, method, args.seeds[0]).get("note") == "N/E"
                     for _ in [0]):
                all_results[ds][method] = {"note": "N/E"}

   
    print(f"\n{'='*80}")
    print(f"TABLE 1 — {args.strategy.upper()} harmful behavior, 20% unreliable clients")
    print(f"{'='*80}")
    header = f"{'Method':<22}"
    for ds in datasets:
        header += f"  {'ACC':>5} {'ASR':>5} {'FPR':>5}"
    print(header)
    print("-" * 80)
    for method in args.methods:
        row = f"{method:<22}"
        for ds in datasets:
            vals = all_results.get(ds, {}).get(method, {})
            if vals.get("note") == "N/E":
                row += f"  {'N/E':>5} {'N/E':>5} {'N/E':>5}"
            else:
                row += (
                    f"  {vals.get('acc', 0.0):>5.1f}"
                    f" {vals.get('asr', 0.0):>5.1f}"
                    f" {vals.get('fpr', 0.0):>5.1f}"
                )
        print(row)

    out_path = Path("results") / "overall_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
