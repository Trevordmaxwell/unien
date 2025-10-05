from __future__ import annotations

from typing import Tuple

import torch


def select_landmarks(M: torch.Tensor, num: int, *, method: str = "uniform") -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose a subset of landmark atoms and return (indices, atoms)."""
    K = M.shape[0]
    if num >= K:
        idx = torch.arange(K, device=M.device)
        return idx, M
    if method == "uniform":
        idx = torch.randperm(K, device=M.device)[:num]
    else:
        scores = M.pow(2).sum(dim=-1)
        _, idx = torch.topk(scores, num)
    return idx, M.index_select(0, idx)


def nystrom_approximation(M: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Return a low-rank approximation using the Nyström method."""
    subset = M.index_select(0, idx)
    gram = subset @ subset.t()
    u, s, _ = torch.linalg.svd(gram, full_matrices=False)
    return u * torch.sqrt(s.clamp_min(1e-6)).unsqueeze(0)


__all__ = ["select_landmarks", "nystrom_approximation"]
