"""
baselines/__init__.py
=====================
Baseline defense registry for TrajRep-FL evaluations.

Stateless:
  fedavg          — FedAvg (McMahan et al., 2017)
  trimmed_mean    — Trimmed Mean (Yin et al., 2018)
  krum            — Krum (Blanchard et al., 2017)

Probabilistic / frequency-domain:
  baybfed         — BayBFed (Kumari et al., 2023)
  flip            — FLIP (Zhang et al., 2023)  [vision only]

Stateful / trajectory-aware:
  fedrecover      — FedRecover (Cao et al., 2023)
  fldetector      — FLDetector (Zhang et al., 2022)
  long_short      — Long-Short History (Gupta et al., 2022)

Proposed:
  trajrep_fl      — TrajRep-FL (this work)

All implement the BaseServer interface:
    aggregate(client_deltas, sampled_client_ids, round_idx, **kwargs)
    → (updated_model, log_dict)
"""

from baselines.all_baselines import (
    FedAvgServer,
    TrimmedMeanServer,
    KrumServer,
    BayBFedServer,
    FedRecoverServer,
    FLIPServer,
    FLDetectorServer,
    LongShortHistoryServer,
    build_server,
    DEFENSE_REGISTRY,
)

__all__ = [
    "FedAvgServer",
    "TrimmedMeanServer",
    "KrumServer",
    "BayBFedServer",
    "FedRecoverServer",
    "FLIPServer",
    "FLDetectorServer",
    "LongShortHistoryServer",
    "build_server",
    "DEFENSE_REGISTRY",
]
