from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .cac import CacheState, apply_cache_penalty
from .control_path import Path
from .energy import total_energy
from .kl_prox import kl_masked_softmax
from .symp_diss_field import BandedField, cde_split_step
from .types import CACCfg, SolverCfg
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

    def __init__(self, cfg: SolverCfg, field: BandedField, controller: Optional[nn.Module] = None, cac_cfg: Optional[CACCfg] = None):
        super().__init__()
        self.cfg = cfg
        self.field = field
        self.controller = controller
        self.cac_cfg = cac_cfg or CACCfg()
        self.register_buffer("_iter", torch.zeros(1, dtype=torch.long))

    def reset(self) -> None:
        self._iter.zero_()

    def _interp(self, start: float, end: float) -> float:
        total = max(self.cfg.T_train, 1)
        step = min(int(self._iter.item()), total)
        alpha = 0.0 if total == 1 else step / (total - 1)
        return float(start + alpha * (end - start))

    def step(self, st: SolverState, X: Path, M: torch.Tensor, cache: CacheState | None = None) -> tuple[SolverState, CacheState | None]:
        self._iter += 1
        device = st.Y.device
        dtype = st.Y.dtype
        beta = torch.tensor(self._interp(self.cfg.beta_start, self.cfg.beta_end), device=device, dtype=dtype)
        tau = torch.tensor(self._interp(self.cfg.tau_start, self.cfg.tau_end), device=device, dtype=dtype)

        if self.controller is not None:
            schedule = self.controller(st)
            beta = schedule.get("beta", beta).to(device=device, dtype=dtype)
            tau = schedule.get("tau", tau).to(device=device, dtype=dtype)

        beta = torch.clamp(beta, min=1e-5)
        tau = torch.clamp(tau, min=1e-5)

        # Y-step
        Y = cde_split_step(st.Y, X, self.field, step=1.0)

        # Cache-as-constraint penalty
        cache_energy = torch.zeros((), device=device, dtype=dtype)
        Y, cache_penalty = apply_cache_penalty(
            Y,
            cache,
            kappa=self.cac_cfg.kappa,
            method=self.cac_cfg.advect,
            decay=self.cac_cfg.decay,
        )
        cache_energy = cache_energy + cache_penalty
        updated_cache = cache.update(Y) if cache is not None else CacheState.from_tensor(Y)

        # Residual & dual signal
        Y_from_P = M_T_P(M, st.Kset, st.P)
        resid = Y - Y_from_P
        Xi = st.Lam + self.cfg.rho * resid

        # Scores per shortlist atom
        T = batched_gather_rows(M, st.Kset)
        scores = torch.bmm(T, Xi.unsqueeze(-1)).squeeze(-1)  # (n,k)

        if self.cfg.use_wmf:
            cost = pairwise_sq_dists(T)
            P_new = wmf_prox_step(st.P, beta * scores, cost=cost, tau=tau, lam_kl=beta)
        else:
            mask = torch.ones_like(st.P, dtype=torch.bool)
            P_new = kl_masked_softmax(st.P, beta * scores, mask=mask)

        Lam_new = st.Lam + self.cfg.rho * (Y - M_T_P(M, st.Kset, P_new))
        energy = total_energy(SolverState(P_new, Y, Lam_new, st.Kset, 0.0), M, self.cfg.rho) + cache_energy
        return SolverState(P=P_new, Y=Y, Lam=Lam_new, Kset=st.Kset, energy=float(energy.detach())), updated_cache


__all__ = ["SolverState", "MirrorPDHG", "batched_gather_rows", "M_T_P", "pairwise_sq_dists"]
