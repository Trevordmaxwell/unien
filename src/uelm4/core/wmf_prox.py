from __future__ import annotations

import torch


def wmf_prox_step(P: torch.Tensor, scores: torch.Tensor, cost: torch.Tensor, tau: float, lam_kl: float = 1.0, num_iters: int = 3) -> torch.Tensor:
    """Compute a lightweight Wasserstein-Mirror Flow prox on a shortlist."""
    if cost.ndim != 3:
        raise ValueError("cost must be (n,k,k)")
    if P.shape != scores.shape:
        raise ValueError("P and scores must match in shape")
    eps = 1e-8
    out = P.clamp_min(eps)
    tau = float(max(tau, 0.0))
    lam = float(max(lam_kl, eps))
    for _ in range(num_iters):
        transport_penalty = torch.bmm(cost, out.unsqueeze(-1)).squeeze(-1)
        logits = torch.log(out) + scores / lam - tau * transport_penalty
        out = torch.softmax(logits, dim=-1)
    return out


__all__ = ["wmf_prox_step"]
