from __future__ import annotations

from typing import Dict

import torch

from .solver_pdhg import SolverState, M_T_P


def compute_primal_dual_residuals(state: SolverState, M: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the constraint residuals used in the augmented Lagrangian."""
    Y_from_P = M_T_P(M, state.Kset, state.P)
    primal = state.Y - Y_from_P
    dual = state.Lam
    return {"primal": primal, "dual": dual}


def compute_energy_terms(state: SolverState, M: torch.Tensor, rho: float) -> Dict[str, torch.Tensor]:
    """Compute individual energy contributions for monitoring."""
    residuals = compute_primal_dual_residuals(state, M)
    energy_primal = 0.5 * residuals["primal"].pow(2).mean()
    energy_dual = 0.5 / max(rho, 1e-6) * residuals["dual"].pow(2).mean()
    entropy = torch.where(state.P > 0, state.P * torch.log(state.P.clamp_min(1e-9)), torch.zeros_like(state.P))
    energy_entropy = entropy.sum(dim=-1).mean()
    return {"primal": energy_primal, "dual": energy_dual, "entropy": energy_entropy}


def total_energy(state: SolverState, M: torch.Tensor, rho: float) -> torch.Tensor:
    terms = compute_energy_terms(state, M, rho)
    return terms["primal"] + terms["dual"] + 1e-2 * terms["entropy"]


__all__ = ["compute_energy_terms", "compute_primal_dual_residuals", "total_energy"]
