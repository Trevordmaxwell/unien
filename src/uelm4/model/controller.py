from __future__ import annotations

import torch
import torch.nn as nn

from ..core.solver_pdhg import SolverState


class MetaController(nn.Module):
    """Predicts solver hyper-parameters from the current state."""

    def __init__(self, beta_range: tuple[float, float], tau_range: tuple[float, float]):
        super().__init__()
        self.beta_range = beta_range
        self.tau_range = tau_range
        self.encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.beta_head = nn.Linear(16, 1)
        self.tau_head = nn.Linear(16, 1)

    def forward(self, state: SolverState) -> dict[str, torch.Tensor]:
        entropy = -torch.sum(state.P * torch.log(state.P.clamp_min(1e-9)), dim=-1).mean()
        dual_norm = state.Lam.norm(dim=-1).mean()
        features = torch.stack([entropy, dual_norm]).unsqueeze(0)
        hidden = self.encoder(features)
        beta_raw = self.beta_head(hidden)
        tau_raw = self.tau_head(hidden)
        beta = self._interpolate(beta_raw, self.beta_range)
        tau = self._interpolate(tau_raw, self.tau_range)
        return {"beta": beta.squeeze(), "tau": tau.squeeze()}

    @staticmethod
    def _interpolate(value: torch.Tensor, rng: tuple[float, float]) -> torch.Tensor:
        lo, hi = rng
        return lo + torch.sigmoid(value) * (hi - lo)


__all__ = ["MetaController"]
