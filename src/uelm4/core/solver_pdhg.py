from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .control_path import Path
from .energy import total_energy
from .kl_prox import kl_masked_softmax
from .preconditioners import IdentityPrecond
from .symp_diss_field import BandedField, cde_split_step
from .types import SolverCfg
from .wmf_prox import wmf_prox_step


@dataclass
class SolverState:
    P: torch.Tensor        # (n,k)
    Y: torch.Tensor        # (n,d)
    Lam: torch.Tensor      # (n,d)
    Kset: torch.Tensor     # (n,k) long indices into M
    energy: float


def batched_gather_rows(M: torch.Tensor, Kset: torch.Tensor) -> torch.Tensor:
    """Gather Kset rows from M for each token returning shape (n,k,d)."""
    n, k = Kset.shape
    idx = Kset.reshape(-1)
    T = M.index_select(0, idx).reshape(n, k, M.shape[1])
    return T


def M_T_P(M: torch.Tensor, Kset: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    T = batched_gather_rows(M, Kset)
    Y = torch.bmm(P.unsqueeze(1), T).squeeze(1)
    return Y


def pairwise_sq_dists(T: torch.Tensor) -> torch.Tensor:
    diff = T.unsqueeze(2) - T.unsqueeze(1)
    return diff.pow(2).sum(dim=-1)


class MirrorPDHG(nn.Module):
    """Phase-A/B solver with KL or WMF proximal updates."""

    def __init__(self, cfg: SolverCfg, field: BandedField, controller: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        self.field = field
        self.precond = IdentityPrecond()
        self.controller = controller
        self.register_buffer("_iter", torch.zeros(1, dtype=torch.long))

    def _interp(self, start: float, end: float) -> float:
        total = max(self.cfg.T_train, 1)
        step = min(int(self._iter.item()), total)
        alpha = 0.0 if total == 1 else step / (total - 1)
        return float(start + alpha * (end - start))

    def step(self, st: SolverState, X: Path, M: torch.Tensor) -> SolverState:
        self._iter += 1
        beta = self._interp(self.cfg.beta_start, self.cfg.beta_end)
        tau = self._interp(self.cfg.tau_start, self.cfg.tau_end)

        if self.controller is not None:
            schedule = self.controller(st)
            beta = schedule.get("beta", beta)
            tau = schedule.get("tau", tau)

        # Y-step
        Y = cde_split_step(st.Y, X, self.field, step=1.0)

        # Residual & dual signal
        Y_from_P = M_T_P(M, st.Kset, st.P)
        resid = Y - Y_from_P
        Xi = st.Lam + self.cfg.rho * resid

        # Scores per shortlist atom
        T = batched_gather_rows(M, st.Kset)
        scores = torch.bmm(T, Xi.unsqueeze(-1)).squeeze(-1)  # (n,k)

        if self.cfg.use_wmf:
            cost = pairwise_sq_dists(T)
            P_new = wmf_prox_step(st.P, beta * scores, cost=cost, tau=max(tau, 1e-4), lam_kl=max(beta, 1e-4))
        else:
            mask = torch.ones_like(st.P, dtype=torch.bool)
            P_new = kl_masked_softmax(st.P, beta * scores, mask=mask)

        Lam_new = st.Lam + self.cfg.rho * (Y - M_T_P(M, st.Kset, P_new))
        energy = float(total_energy(SolverState(P_new, Y, Lam_new, st.Kset, 0.0), M, self.cfg.rho))
        return SolverState(P=P_new, Y=Y, Lam=Lam_new, Kset=st.Kset, energy=energy)


__all__ = ["SolverState", "MirrorPDHG", "batched_gather_rows", "M_T_P", "pairwise_sq_dists"]
