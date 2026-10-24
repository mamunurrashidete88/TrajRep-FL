
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

ABLATION_VARIANTS = [
    "no_encoder",
    "ema_encoder",
    "no_multi_res",
    "no_interaction",
    "no_recovery",
    "full",              
]

VARIANT_LABELS = {
    "no_encoder":     "w/o Trajectory Encoder",
    "ema_encoder":    "EMA Trajectory Summary",
    "no_multi_res":   "w/o Multi-Resolution",
    "no_interaction": "w/o Interaction-Aware Weighting",
    "no_recovery":    "w/o Influence Recovery",
    "full":           "TrajRep-FL (full)",
}


def run_variant(
    dataset: str,
    variant: str,
    seed: int,
    strategy: str = "gradual_drift",
) -> Dict:
    
    cmd = [
        sys.executable,
        "experiments/run_experiment.py",
        "--dataset", dataset,
        "--method", "trajrep_fl",
        "--strategy", strategy,
        "--seed", str(seed),
    ]
    if variant != "full":
        cmd += ["--ablation", variant]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
    if result.returncode != 0:
        print(f"[WARN] {variant}/{dataset}/seed{seed} failed:\n{result.stderr[-500:]}")
        return {}

    
    run_tag = (
        f"{dataset}_trajrep_fl_{strategy}"
        + (f"_{variant}" if variant != "full" else "")
        + f"_seed{seed}"
    )
    results_path = Path("results") / dataset / run_tag / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="TrajRep-FL Ablation Study")
    parser.add_argument("--dataset", default="femnist",
                        choices=["femnist", "cifar10", "shakespeare", "all"])
    parser.add_argument("--strategy", default="gradual_drift",
                        choices=["gradual_drift", "static"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[11, 22, 33, 44, 55])
    parser.add_argument("--variants", nargs="+", default=ABLATION_VARIANTS,
                        choices=ABLATION_VARIANTS)
    args = parser.parse_args()

    datasets = (
        ["femnist", "cifar10", "shakespeare"]
        if args.dataset == "all"
        else [args.dataset]
    )

    all_results: Dict = {}

    for ds in datasets:
        all_results[ds] = {}
        for variant in args.variants:
            variant_results = []
            for seed in args.seeds:
                print(f"[ablation] {ds} | {VARIANT_LABELS[variant]} | seed={seed}")
                r = run_variant(ds, variant, seed, args.strategy)
                if r:
                    variant_results.append(r.get("final", {}))

            if variant_results:
                # Average across seeds
                avg = {}
                keys = ["acc", "asr", "fpr"]
                for k in keys:
                    vals = [v[k] for v in variant_results if k in v]
                    if vals:
                        avg[k] = sum(vals) / len(vals)
                all_results[ds][VARIANT_LABELS[variant]] = avg

    
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    for ds in datasets:
        print(f"\n{ds.upper()}")
        print(f"{'Configuration':<40} {'ACC':>6} {'ASR':>6} {'FPR':>6}")
        print("-" * 62)
        for variant in args.variants:
            label = VARIANT_LABELS[variant]
            vals = all_results[ds].get(label, {})
            acc = f"{vals.get('acc', 0.0):.1f}"
            asr = f"{vals.get('asr', 0.0):.1f}"
            fpr = f"{vals.get('fpr', 0.0):.1f}"
            print(f"{label:<40} {acc:>6} {asr:>6} {fpr:>6}")

    out_path = Path("results") / "ablation_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
