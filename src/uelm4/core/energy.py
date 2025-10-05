from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:
    from .solver_pdhg import SolverState


def _batched_gather_rows(M: torch.Tensor, Kset: torch.Tensor) -> torch.Tensor:
    n, k = Kset.shape
    idx = Kset.reshape(-1)
    return M.index_select(0, idx).reshape(n, k, M.shape[1])


def _m_t_p(M: torch.Tensor, Kset: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    gathered = _batched_gather_rows(M, Kset)
    return torch.bmm(P.unsqueeze(1), gathered).squeeze(1)


def compute_primal_dual_residuals(state: "SolverState", M: torch.Tensor) -> dict[str, torch.Tensor]:
    Y_from_P = _m_t_p(M, state.Kset, state.P)
    primal = state.Y - Y_from_P
    dual = state.Lam
    return {"primal": primal, "dual": dual}


def compute_energy_terms(state: "SolverState", M: torch.Tensor, rho: float) -> Dict[str, torch.Tensor]:
    residuals = compute_primal_dual_residuals(state, M)
    energy_primal = 0.5 * residuals["primal"].pow(2).mean()
    energy_dual = 0.5 / max(rho, 1e-6) * residuals["dual"].pow(2).mean()
    entropy = -torch.sum(state.P * torch.log(state.P.clamp_min(1e-9)), dim=-1).mean()
    return {"primal": energy_primal, "dual": energy_dual, "entropy": entropy}


def total_energy(state: "SolverState", M: torch.Tensor, rho: float) -> torch.Tensor:
    terms = compute_energy_terms(state, M, rho)
    return terms["primal"] + terms["dual"] + 1e-2 * terms["entropy"]


__all__ = ["compute_energy_terms", "compute_primal_dual_residuals", "total_energy"]
