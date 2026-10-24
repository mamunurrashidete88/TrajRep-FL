
from __future__ import annotations
from typing import Dict, List, Optional
import math
import torch
import torch.nn.functional as F


def reconstruction_loss(x_hat: Dict[int, torch.Tensor], x_target: Dict[int, torch.Tensor], client_ids: List[int]) -> torch.Tensor:
    terms = [torch.sum((x_hat[i] - x_target[i].detach()) ** 2) for i in client_ids if i in x_hat and i in x_target]
    if terms:
        return torch.stack(terms).mean()
    return next(iter(x_target.values())).new_zeros(()) if x_target else torch.tensor(0.)


def smoothness_loss(alpha_current: torch.Tensor, client_ids: List[int], alpha_previous: Dict[int, Optional[float]]) -> torch.Tensor:
    terms = [(alpha_current[j] - float(alpha_previous[i])) ** 2 for j, i in enumerate(client_ids) if alpha_previous.get(i) is not None]
    return torch.stack(terms).mean() if terms else alpha_current.new_zeros(())


def entropy_loss(alpha_current: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    
    n = alpha_current.numel()
    if n < 2:
        return alpha_current.new_zeros(())
    return (alpha_current * torch.log((n * alpha_current).clamp_min(torch.finfo(alpha_current.dtype).tiny))).sum() / math.log(n)


def reliability_target(u_matrix: torch.Tensor, rec_err: Optional[torch.Tensor] = None, kappa: float = 2.0,
                       w_tsc: float = 1.0, w_psc: float = 1.0, w_mr: float = 1.0,
                       w_rec: float = 1.0, epsilon_cal: float = 1e-4) -> torch.Tensor:

    if u_matrix.numel() == 0:
        return u_matrix.new_zeros((0,))
    with torch.no_grad():
        a = (u_matrix[:, 0] + u_matrix[:, 1] + u_matrix[:, 2]) / 6.0
        return (1.0 - a).clamp(epsilon_cal, 1.0 - epsilon_cal).detach()


def reliability_separation_loss(q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    if q_pred.numel() == 0:
        return q_pred.new_zeros(())
    return F.binary_cross_entropy(q_pred, q_target.detach())


def trajectory_loss(x_hat: Dict[int, torch.Tensor], x_target: Dict[int, torch.Tensor], client_ids: List[int],
                    alpha_current: torch.Tensor, alpha_client_ids: List[int],
                    alpha_previous: Dict[int, Optional[float]], q_pred: torch.Tensor, q_target: torch.Tensor,
                    lambda_smooth: float = .1, lambda_ent: float = .01, lambda_rel: float = 1.0) -> torch.Tensor:
    return (reconstruction_loss(x_hat, x_target, client_ids)
            + lambda_rel * reliability_separation_loss(q_pred, q_target)
            + lambda_smooth * smoothness_loss(alpha_current, alpha_client_ids, alpha_previous)
            + lambda_ent * entropy_loss(alpha_current))
