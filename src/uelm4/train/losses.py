from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..core.solver_pdhg import SolverState


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss."""
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))


def energy_regulariser(state: SolverState, weight: float = 1e-3) -> torch.Tensor:
    energy = torch.tensor(state.energy, device=state.Y.device, dtype=state.Y.dtype)
    return weight * energy


def total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    state: Optional[SolverState] = None,
    weights: Dict[str, float] | None = None,
) -> torch.Tensor:
    weights = weights or {}
    loss = language_model_loss(logits, targets)
    if state is not None and weights.get("energy", 0.0) > 0:
        loss = loss + energy_regulariser(state, weights["energy"])
    return loss


__all__ = ["language_model_loss", "energy_regulariser", "total_loss"]
