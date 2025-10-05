from __future__ import annotations

import torch
import torch.nn as nn

from ..core.solver_pdhg import M_T_P, batched_gather_rows


class Scout(nn.Module):
    """Amortised initializer for the simplex state."""

    def __init__(self, d: int):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, embeddings: torch.Tensor, memory: torch.Tensor, Kset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.project(embeddings)  # (n,d)
        gathered = batched_gather_rows(memory, Kset)  # (n,k,d)
        scores = torch.bmm(gathered, queries.unsqueeze(-1)).squeeze(-1)
        P0 = torch.softmax(scores, dim=-1)
        Y0 = M_T_P(memory, Kset, P0)
        return P0, Y0


__all__ = ["Scout"]
