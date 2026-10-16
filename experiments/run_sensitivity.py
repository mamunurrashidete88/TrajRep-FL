"""
experiments/run_sensitivity.py
================================
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))


GAMMA_GRID    = [0.5, 0.7, 0.9, 0.95]   
N_W_GRID      = [2, 5, 10, 20]           


def run_with_override(
    dataset: str,
    seed: int,
    strategy: str,
    extra_yaml_overrides: Dict,
) -> Dict:
    """Write a temporary config override and run an experiment."""
    import tempfile, yaml

    config_path = Path(__file__).parent.parent / "configs" / f"{dataset}_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    for k, v in extra_yaml_overrides.items():
        cfg["trajrep_fl"][k] = v
    cfg["adversary"]["strategy"] = strategy

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as tmp:
        yaml.dump(cfg, tmp)
        tmp_path = tmp.name

    cmd = [
        sys.executable,
        "experiments/run_experiment.py",
        "--dataset", dataset,
        "--method", "trajrep_fl",
        "--strategy", strategy,
        "--seed", str(seed),
        "--config", tmp_path,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent)
    )
    Path(tmp_path).unlink(missing_ok=True)

    tag = (
        f"{dataset}_trajrep_fl_{strategy}_seed{seed}"
    )
    results_path = Path("results") / dataset / tag / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f).get("final", {})
    return {}


def sweep_gamma(datasets, seeds, strategy="gradual_drift"):
    results = {}
    for ds in datasets:
        results[ds] = {}
        for g in GAMMA_GRID:
            vals = []
            for s in seeds:
                print(f"[sensitivity] gamma={g} | {ds} | seed={s}")
                r = run_with_override(ds, s, strategy, {"gamma": g})
                if r:
                    vals.append(r)
            if vals:
                avg = {k: sum(v[k] for v in vals if k in v) / len(vals)
                       for k in ["acc", "asr", "fpr"]}
                results[ds][f"gamma={g}"] = avg
    return results


def sweep_nw(datasets, seeds, strategy="gradual_drift"):
    results = {}
    for ds in datasets:
        results[ds] = {}
        for nw in N_W_GRID:
            vals = []
            for s in seeds:
                print(f"[sensitivity] N_w={nw} | {ds} | seed={s}")
                r = run_with_override(ds, s, strategy, {"recovery_window": nw})
                if r:
                    vals.append(r)
            if vals:
                avg = {k: sum(v[k] for v in vals if k in v) / len(vals)
                       for k in ["acc", "asr", "fpr"]}
                results[ds][f"N_w={nw}"] = avg
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="femnist",
                        choices=["femnist", "cifar10", "shakespeare", "all"])
    parser.add_argument("--param", default="gamma",
                        choices=["gamma", "N_w", "all"])
    parser.add_argument("--strategy", default="gradual_drift")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[11, 22, 33, 44, 55])
    args = parser.parse_args()

    datasets = (
        ["femnist", "cifar10", "shakespeare"]
        if args.dataset == "all" else [args.dataset]
    )

    all_results = {}
    if args.param in ("gamma", "all"):
        all_results["gamma"] = sweep_gamma(datasets, args.seeds, args.strategy)
    if args.param in ("N_w", "all"):
        all_results["N_w"] = sweep_nw(datasets, args.seeds, args.strategy)

    out_path = Path("results") / "sensitivity_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSensitivity results saved → {out_path}")

    # Print tables
    for param, res in all_results.items():
        print(f"\n{'='*60}\nSensitivity: {param}\n{'='*60}")
        for ds, ds_res in res.items():
            print(f"\n{ds.upper()}")
            print(f"{'Value':<20} {'ACC':>6} {'ASR':>6} {'FPR':>6}")
            print("-" * 40)
            for val_key, metrics in ds_res.items():
                print(
                    f"{val_key:<20} "
                    f"{metrics.get('acc', 0.0):>6.1f} "
                    f"{metrics.get('asr', 0.0):>6.1f} "
                    f"{metrics.get('fpr', 0.0):>6.1f}"
                )


if __name__ == "__main__":
    main()
