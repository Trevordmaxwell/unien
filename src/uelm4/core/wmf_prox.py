from __future__ import annotations

import torch


def _ensure_tensor(value: float | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.tensor(float(value), device=like.device, dtype=like.dtype)


def wmf_prox_step(
    P: torch.Tensor,
    scores: torch.Tensor,
    cost: torch.Tensor,
    tau: float | torch.Tensor,
    lam_kl: float | torch.Tensor = 1.0,
    num_iters: int = 3,
    eps: float | torch.Tensor = 1e-3,
    cost_scale: float | torch.Tensor = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a lightweight Wasserstein-Mirror Flow prox on a shortlist."""
    if cost.ndim != 3:
        raise ValueError("cost must be (n,k,k)")
    if P.shape != scores.shape:
        raise ValueError("P and scores must match in shape")
    tau_t = torch.clamp(_ensure_tensor(tau, P), min=1e-5)
    lam_t = torch.clamp(_ensure_tensor(lam_kl, P), min=1e-5)
    eps_t = torch.clamp(_ensure_tensor(eps, P), min=1e-8)
    scale_t = _ensure_tensor(cost_scale, P)
    out = P.clamp_min(eps_t)
    cost_scaled = cost * scale_t
    if mask is not None:
        mask = mask.to(dtype=torch.bool, device=P.device)
    for _ in range(num_iters):
        transport_penalty = torch.bmm(cost_scaled, out.unsqueeze(-1)).squeeze(-1)
        logits = torch.log(out.clamp_min(eps_t)) + scores / lam_t - tau_t * transport_penalty
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        out = torch.softmax(logits, dim=-1)
        out = out.clamp_min(eps_t)
    out = out / out.sum(dim=-1, keepdim=True)
    return out


__all__ = ["wmf_prox_step"]
