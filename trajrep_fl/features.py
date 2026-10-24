

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Round-level 
# ---------------------------------------------------------------------------

def compute_round_mean(
    flat_deltas: Dict[int, torch.Tensor],
    client_ids: List[int],
) -> torch.Tensor:

    vecs = [flat_deltas[cid] for cid in client_ids if cid in flat_deltas]
    if not vecs:
        return torch.zeros(1)
    total = vecs[0].clone()
    for v in vecs[1:]:
        total.add_(v)
    return total.div_(len(vecs))


def compute_latent_centroid(
    embeddings: Dict[int, torch.Tensor],
    client_ids: List[int],
) -> torch.Tensor:

    zs = [embeddings[cid] for cid in client_ids if cid in embeddings]
    if not zs:
        return torch.zeros(64)
    return torch.stack(zs).mean(dim=0)


# ---------------------------------------------------------------------------
# Float builder 
# ---------------------------------------------------------------------------

def build_feature_vector(
    client_id: int,
    z: torch.Tensor,
    flat_delta: torch.Tensor,
    d_single: float,
    d_fast: float,
    d_slow: float,
    centroid: torch.Tensor,
    round_mean_delta: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:

    delta_mr = abs(d_fast - d_slow)
    tsc = beta * d_single + (1.0 - beta) * delta_mr

    z_cpu = z.detach().cpu().float()
    c_cpu = centroid.detach().cpu().float()
    z_norm = F.normalize(z_cpu.unsqueeze(0), dim=-1)
    c_norm = F.normalize(c_cpu.unsqueeze(0), dim=-1)
    psc = 1.0 - (z_norm * c_norm).sum().item()

    delta_norm = flat_delta.detach().float().norm().item()

    d_cpu = flat_delta.detach().cpu().float()
    g_cpu = round_mean_delta.detach().cpu().float()
    d_n = F.normalize(d_cpu.unsqueeze(0), dim=-1)
    g_n = F.normalize(g_cpu.unsqueeze(0), dim=-1)
    cos_align = (d_n * g_n).sum().item()

    return torch.tensor(
        [tsc, psc, delta_mr, delta_norm, cos_align],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Differentiable builder 
# ---------------------------------------------------------------------------

def _cos_distance_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
   
    a_n = F.normalize(a.unsqueeze(0), dim=-1)
    b_n = F.normalize(b.unsqueeze(0), dim=-1)
    return 1.0 - (a_n * b_n).sum()


def build_feature_vector_tensor(
    z: torch.Tensor,
    flat_delta: torch.Tensor,
    anchor_single: torch.Tensor | None,
    anchor_fast: torch.Tensor | None,
    anchor_slow: torch.Tensor | None,
    centroid: torch.Tensor,
    round_mean_delta: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    
    device = z.device

    # --- anchors as detached references ---
    if anchor_single is None:
        
        zero = z.sum() * 0.0  
        d_single = zero
        d_fast = zero
        d_slow = zero
    else:
        a_s = anchor_single.detach().to(device)
        a_f = anchor_fast.detach().to(device)
        a_l = anchor_slow.detach().to(device)
        d_single = _cos_distance_t(z, a_s)
        d_fast = _cos_distance_t(z, a_f)
        d_slow = _cos_distance_t(z, a_l)

    delta_mr = torch.abs(d_fast - d_slow)
    tsc = beta * d_single + (1.0 - beta) * delta_mr

    # PSC 
    c = centroid.detach().to(device)
    psc = _cos_distance_t(z, c)

    # Update geometry 
    with torch.no_grad():
        delta_norm = flat_delta.float().norm().to(device)
        d_n = F.normalize(flat_delta.float().unsqueeze(0), dim=-1)
        g_n = F.normalize(round_mean_delta.float().unsqueeze(0), dim=-1)
        cos_align = (d_n * g_n).sum().to(device)

    u = torch.stack([
        tsc,
        psc,
        delta_mr,
        delta_norm,
        cos_align,
    ])
    return u
