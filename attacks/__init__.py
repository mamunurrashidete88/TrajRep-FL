"""
attacks/__init__.py
===================
Attack registry and factory for TrajRep-FL evaluations.

Payloads
--------
backdoor       — fixed pixel-trigger backdoor (FEMNIST, CIFAR-10)
backdoor_token — fixed token-trigger backdoor (Shakespeare)
wanet          — elastic image-warping stealthy trigger (vision only)
narcissus      — clean-label adversarial perturbation (vision only)
label_flip     — untargeted label-flipping (utility stress-test)

Execution strategies
--------------------
static         — inject payload from round 0 (non-adaptive baseline)
delayed_shift  — benign for delay_rounds=30, then abruptly activate
gradual_drift  — linear interpolation toward harmful over drift_duration=50
score_aware    — white-box PGD with trajectory-budget constraint (strongest)
burst          — alternating benign/malicious windows (intermediate case)
"""

from attacks.base_attack import BaseAttack
from attacks.backdoor import BackdoorAttack, LabelFlipAttack, TokenBackdoorAttack
from attacks.delayed_shift import DelayedShiftAttack
from attacks.gradual_drift import GradualDriftAttack, BurstAttack
from attacks.score_aware import ScoreAwareAdaptiveAttack
from attacks.wanet import WaNetAttack, NARCISSUSAttack


def build_attack(
    strategy: str,
    payload: str,
    adversarial_client_ids,
    cfg: dict,
    device,
    score_fn=None,
) -> BaseAttack:
    """
    Construct an attack from a (strategy, payload) pair.

    strategy : "static" | "delayed_shift" | "gradual_drift" | "score_aware" | "burst"
    payload  : "backdoor" | "backdoor_token" | "wanet" | "narcissus" | "label_flip"
    """
    adv_cfg = cfg.get("adversary", cfg)
    target_class = adv_cfg.get("target_class", 0)

    # ── Payload ───────────────────────────────────────────────────────
    if payload == "label_flip":
        base = LabelFlipAttack(
            adversarial_client_ids=list(adversarial_client_ids),
            source_class=adv_cfg.get("source_class", 1),
            target_class=target_class,
            device=device,
        )
    elif payload == "backdoor":
        base = BackdoorAttack(
            adversarial_client_ids=list(adversarial_client_ids),
            target_class=target_class,
            trigger_size=adv_cfg.get("trigger_size", 3),
            trigger_position=adv_cfg.get("trigger_position", "bottom_right"),
            scale=adv_cfg.get("backdoor_scale", 10.0),
            device=device,
        )
    elif payload == "backdoor_token":
        base = TokenBackdoorAttack(
            adversarial_client_ids=list(adversarial_client_ids),
            target_class=target_class,
            trigger_length=adv_cfg.get("trigger_length", 5),
            device=device,
        )
    elif payload == "wanet":
        base = WaNetAttack(
            adversarial_client_ids=list(adversarial_client_ids),
            target_class=target_class,
            device=device,
        )
    elif payload == "narcissus":
        base = NARCISSUSAttack(
            adversarial_client_ids=list(adversarial_client_ids),
            target_class=target_class,
            device=device,
        )
    else:
        raise ValueError(f"Unknown payload: '{payload}'")

    # ── Execution strategy ────────────────────────────────────────────
    if strategy == "static":
        return base

    if strategy == "delayed_shift":
        return DelayedShiftAttack(
            payload_attack=base,
            adversarial_client_ids=list(adversarial_client_ids),
            delay_rounds=adv_cfg.get("delay_rounds", 30),
            device=device,
        )

    if strategy == "gradual_drift":
        return GradualDriftAttack(
            adversarial_client_ids=list(adversarial_client_ids),
            drift_duration=adv_cfg.get("drift_duration", 50),
            target_class=target_class,
            device=device,
        )

    if strategy == "score_aware":
        return ScoreAwareAdaptiveAttack(
            adversarial_client_ids=list(adversarial_client_ids),
            score_fn=score_fn,
            stealth_budget=adv_cfg.get("stealth_budget", 0.15),
            pgd_steps=adv_cfg.get("pgd_steps", 20),
            pgd_lr=adv_cfg.get("pgd_lr", 0.01),
            target_class=target_class,
            device=device,
        )

    if strategy == "burst":
        return BurstAttack(
            payload_attack=base,
            adversarial_client_ids=list(adversarial_client_ids),
            benign_window=adv_cfg.get("burst_benign_window", 5),
            malicious_window=adv_cfg.get("burst_malicious_window", 5),
            device=device,
        )

    raise ValueError(f"Unknown attack strategy: '{strategy}'")


__all__ = [
    "BaseAttack",
    "BackdoorAttack",
    "LabelFlipAttack",
    "TokenBackdoorAttack",
    "DelayedShiftAttack",
    "GradualDriftAttack",
    "BurstAttack",
    "ScoreAwareAdaptiveAttack",
    "WaNetAttack",
    "NARCISSUSAttack",
    "build_attack",
]
