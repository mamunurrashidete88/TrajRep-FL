

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loaders import build_federated_datasets, load_dataset
from fl.client import FLClient, ShakespeareClient
from fl.federated_runner import FederatedRunner
from models import build_model
from utils.logging_utils import setup_logging
from utils.reproducibility import seed_everything
from evaluation.evaluator import FederatedEvaluator
from attacks import build_attack

from baselines.all_baselines import (
    FedAvgServer,
    TrimmedMeanServer,
    KrumServer,
    BayBFedServer,
    FedRecoverServer,
    FLIPServer,
    LongShortHistoryServer,
    FLDetectorServer,
)
from trajrep_fl.server import TrajRepFLServer


METHOD_MAP = {
    "trajrep_fl":   TrajRepFLServer,
    "fedavg":       FedAvgServer,
    "trimmed_mean": TrimmedMeanServer,
    "krum":         KrumServer,
    "baybfed":      BayBFedServer,
    "fedrecover":   FedRecoverServer,
    "flip":         FLIPServer,
    "long_short":   LongShortHistoryServer,
    "fldetector":   FLDetectorServer,
}

ABLATION_FLAGS = {
    "no_encoder":     "ablation_no_encoder",
    "ema_encoder":    "ablation_ema_encoder",
    "no_multi_res":   "ablation_no_multi_res",
    "no_interaction": "ablation_no_interaction",
    "no_recovery":    "ablation_no_recovery",
    "temporal_only":  "ablation_temporal_only",
    "peer_only":      "ablation_peer_only",
    "fixed_fusion":   "ablation_fixed_fusion",
}


def load_config(dataset: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).parent.parent / "configs" / f"{dataset}_config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def build_clients(
    client_datasets,
    fl_cfg: dict,
    ds_name: str,
    device: torch.device,
) -> List:
    """Wrap each client dataset in FLClient / ShakespeareClient."""
    is_text = ds_name == "shakespeare"
    clients = []
    for cid, ds in enumerate(client_datasets):
        cls = ShakespeareClient if is_text else FLClient
        clients.append(cls(
            client_id=cid,
            dataset=ds,
            num_classes=fl_cfg.get("num_classes", 62),
            local_epochs=fl_cfg.get("local_epochs", 1),
            local_batch_size=fl_cfg.get("local_batch_size", 32),
            learning_rate=fl_cfg.get("learning_rate", 0.01),
            momentum=fl_cfg.get("momentum", 0.9),
            device=device,
        ))
    return clients


def build_backdoor_loader(test_dataset, adv_cfg: dict, dataset_name: str,
                          model_cfg: dict, device: torch.device) -> DataLoader:
    import copy, torch
    from torch.utils.data import TensorDataset

    if adv_cfg.get('payload') == 'label_flip':
        return None  # label flipping has no targeted trigger ASR
    try:
        from attacks.backdoor import BackdoorAttack, TokenBackdoorAttack
        target_class = adv_cfg.get("target_class", 0)

        if dataset_name in ("femnist", "cifar10"):
            atk = BackdoorAttack(
                adversarial_client_ids=[],
                target_class=target_class,
                trigger_size=adv_cfg.get("trigger_size", 3),
                trigger_position=adv_cfg.get("trigger_position", "bottom_right"),
                scale=1.0,
                device=device,
            )
            images, labels = [], []
            for x, y in DataLoader(test_dataset, batch_size=256):
                mask = y != target_class  
                if not bool(mask.any()):
                    continue
                x_bd = atk.add_trigger_to_image(x[mask])
                images.append(x_bd)
                labels.append(torch.full((len(x_bd),), target_class, dtype=torch.long))
            bd_ds = TensorDataset(torch.cat(images), torch.cat(labels))

        elif dataset_name == "shakespeare":
            atk = TokenBackdoorAttack(
                adversarial_client_ids=[],
                target_class=target_class,
                trigger_length=adv_cfg.get("trigger_length", 5),
                device=device,
            )
            seqs, tgts = [], []
            for x, _ in DataLoader(test_dataset, batch_size=256):
                x_bd = x.clone()
                x_bd[:, :atk.trigger_length] = 1
                seqs.append(x_bd)
                tgts.append(torch.full_like(x_bd, target_class, dtype=torch.long))
            bd_ds = TensorDataset(torch.cat(seqs), torch.cat(tgts))
        else:
            bd_ds = test_dataset

        return DataLoader(bd_ds, batch_size=256, shuffle=False)
    except Exception as e:
        raise RuntimeError("Unable to construct attack-specific evaluation loader; refusing to report clean accuracy as ASR") from e


def build_server(method: str, model, model_dim: int, cfg: dict,
                 device: torch.device, ablation: Optional[str] = None):
    fl_cfg  = cfg["federated"]
    lr      = fl_cfg.get("server_learning_rate", 1.0)   # paper §B.3
    n_cl    = fl_cfg["num_clients"]
    cprf    = fl_cfg.get("clients_per_round_fraction", 0.1)

    if method == "trajrep_fl":
        traj_cfg = dict(cfg.get("trajrep_fl", {}))
        traj_cfg["num_clients"] = n_cl
        if ablation and ablation in ABLATION_FLAGS:
            traj_cfg[ABLATION_FLAGS[ablation]] = True

        return TrajRepFLServer(
            model_dim=model_dim,
            cfg=traj_cfg,
            device=device,
            global_model=model,
            learning_rate=lr,
            num_clients=n_cl,
            clients_per_round_fraction=cprf,
        )

    ServerClass = METHOD_MAP[method]
    kwargs: Dict[str, Any] = dict(
        global_model=model,
        learning_rate=lr,
        device=device,
        num_clients=n_cl,
        clients_per_round_fraction=cprf,
    )
    if method == "trimmed_mean":
        kwargs["trim_fraction"] = cfg.get("baselines", {}).get("trim_fraction", 0.1)
    elif method == "krum":
        kwargs["num_to_select"] = cfg.get("baselines", {}).get("krum_select", 1)

    return ServerClass(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="TrajRep-FL Experiment Runner")
    parser.add_argument("--dataset",   default="femnist",
                        choices=["femnist", "cifar10", "shakespeare"])
    parser.add_argument("--method",    default="trajrep_fl",
                        choices=list(METHOD_MAP.keys()))
    parser.add_argument("--strategy",  default="static",
                        choices=["static", "delayed_shift", "gradual_drift",
                                 "score_aware", "burst"])
    parser.add_argument("--payload",   default=None)
    parser.add_argument("--ablation",  default=None,
                        choices=list(ABLATION_FLAGS.keys()))
    parser.add_argument("--seed",      type=int, default=None)
    parser.add_argument("--rounds",    type=int, default=None)
    parser.add_argument("--adv_frac",  type=float, default=None)
    parser.add_argument("--participation-rate", type=float, default=None,
                        help="Override clients_per_round_fraction; report this setting with results")
    parser.add_argument("--config",    default=None)
    parser.add_argument("--output_dir",default=None)
    parser.add_argument("--no_cuda",   action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Explicitly allow synthetic fallback data; results are NOT scientific evidence")
    args = parser.parse_args()

    cfg = load_config(args.dataset, args.config)
    exp_cfg = cfg["experiment"]
    fl_cfg  = cfg["federated"]
    ds_cfg  = cfg["dataset"]
    model_cfg = cfg["model"]
    adv_cfg = cfg["adversary"]
    eval_cfg = cfg.get("evaluation", {})

    if args.seed is not None:
        exp_cfg["seed"] = args.seed
    if args.rounds is not None:
        fl_cfg["num_rounds"] = args.rounds
    if args.adv_frac is not None:
        adv_cfg["fraction"] = args.adv_frac
    if args.participation_rate is not None:
        if not 0 < args.participation_rate <= 1:
            parser.error('--participation-rate must be in (0,1]')
        fl_cfg['clients_per_round_fraction'] = args.participation_rate
    adv_cfg["strategy"] = args.strategy
    if args.payload:
        adv_cfg["payload"] = args.payload

    fl_cfg["num_classes"] = model_cfg.get("num_classes", ds_cfg.get("num_classes", 62))

    output_dir = Path(args.output_dir or exp_cfg.get("output_dir", "results"))
    run_tag = (
        f"{args.dataset}_{args.method}_{args.strategy}"
        + (f"_{args.ablation}" if args.ablation else "")
        + f"_seed{exp_cfg['seed']}"
    )
    output_dir = output_dir / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_dir / "run.log"))
    logger.info(f"TrajRep-FL | {run_tag}")
    if args.synthetic_smoke:
        logger.warning('SYNTHETIC SMOKE MODE: evaluation results are not manuscript evidence')

    seed_everything(exp_cfg["seed"])

    device = torch.device("cpu") if (args.no_cuda or not torch.cuda.is_available()) else torch.device("cuda")
    logger.info(f"Device: {device}")

    # ── Datasets ─────────────────────────────────────────────────────
    client_datasets, test_dataset = build_federated_datasets(
        dataset_name=ds_cfg["name"],
        data_path=ds_cfg["path"],
        num_clients=fl_cfg["num_clients"],
        non_iid_type=ds_cfg.get("non_iid_type", "natural"),
        dirichlet_alpha=ds_cfg.get("dirichlet_alpha", 0.3),
        seed=exp_cfg["seed"],
        allow_synthetic=args.synthetic_smoke,
    )
    logger.info(f"Loaded {len(client_datasets)} client datasets")

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    backdoor_loader = build_backdoor_loader(
        test_dataset, adv_cfg, ds_cfg["name"], model_cfg, device
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(model_cfg).to(device)
    model_dim = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_cfg['name']} | {model_dim:,} parameters")

    # ── Clients ───────────────────────────────────────────────────────
    clients = build_clients(client_datasets, fl_cfg, ds_cfg["name"], device)

    # ── Adversarial clients ───────────────────────────────────────────
    adv_frac = adv_cfg["fraction"]
    n_clients = fl_cfg["num_clients"]
    rng = torch.Generator()
    rng.manual_seed(exp_cfg["seed"])
    adv_ids = sorted(
        torch.randperm(n_clients, generator=rng)[: int(adv_frac * n_clients)].tolist()
    )
    logger.info(f"Adversarial clients: {len(adv_ids)} / {n_clients}")

    attack = build_attack(
        strategy=adv_cfg["strategy"],
        payload=adv_cfg["payload"],
        adversarial_client_ids=adv_ids,
        cfg=cfg,
        device=device,
    )

    # ── Server ────────────────────────────────────────────────────────
    server = build_server(args.method, model, model_dim, cfg, device, args.ablation)

    # ── Evaluator ─────────────────────────────────────────────────────
    evaluator = FederatedEvaluator(
        test_loader=test_loader,
        backdoor_test_loader=backdoor_loader,
        num_classes=model_cfg.get("num_classes", 62),
        target_class=adv_cfg.get("target_class", 0),
        target_acc_threshold=eval_cfg.get("target_acc_threshold", 60.0),
        device=device,
    )

    # ── Federated Runner ──────────────────────────────────────────────
    runner = FederatedRunner(
        global_model=model,
        server=server,
        clients=clients,
        adversarial_clients=adv_ids,
        attack=attack,
        test_loader=test_loader,
        backdoor_test_loader=backdoor_loader,
        cfg=cfg,
        device=device,
        target_acc_threshold=eval_cfg.get("target_acc_threshold", 60.0),
    )

    # ── Run ───────────────────────────────────────────────────────────
    t0 = time.time()
    metrics = runner.run()
    elapsed = time.time() - t0
    logger.info(f"Training complete in {elapsed:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────
    summary = metrics.summary() if hasattr(metrics, "summary") else {}
    final_metrics = {
        "acc":  summary.get("acc",  0.0),
        "asr":  summary.get("asr",  0.0),
        "fpr":  summary.get("fpr",  0.0),
    }
    results = {
        "run_tag":   run_tag,
        "elapsed_s": elapsed,
        "final":     final_metrics,
        "history":   summary,
        "config":    cfg,
        "synthetic_smoke": args.synthetic_smoke,
        "data_provenance": "synthetic permitted" if args.synthetic_smoke else "real data required",
    }

    def _last(key):
        return next((float(row[key]) for row in reversed(runner.round_logs)
                     if row.get(key) is not None), None)
    results['final'] = {'acc': _last('acc'), 'asr': _last('asr'), 'fpr': _last('fpr')}
    results['summary'] = summary
    final_metrics = results['final']

    def _jsonable(v):
        if isinstance(v, dict):
            return {str(k): _jsonable(z) for k, z in v.items()}
        if isinstance(v, (list, tuple)):
            return [_jsonable(z) for z in v]
        if torch.is_tensor(v):
            return v.detach().cpu().tolist()
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        return str(v)

    with open(output_dir / 'round_logs.jsonl', 'w') as f:
        for row in runner.round_logs:
            f.write(json.dumps(_jsonable(row)) + '\n')

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved → {results_path}")

    print(f"\n{'='*60}")
    print(f"  Method   : {args.method}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Strategy : {args.strategy}")
    print(f"  ACC      : {(final_metrics['acc'] or 0.0):.2f}%")
    print(f"  ASR      : {str(round(final_metrics['asr'], 2)) + '%' if final_metrics['asr'] is not None else 'N/A'}")
    print(f"  FPR      : {(final_metrics['fpr'] or 0.0):.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
