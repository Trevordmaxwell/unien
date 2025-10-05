from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from .ann import ANNIndex, load_ann_index


def _memory_table(memory: torch.Tensor | nn.Module) -> torch.Tensor:
    if hasattr(memory, "landmarks_view"):
        return getattr(memory, "landmarks_view")()
    if isinstance(memory, nn.Parameter):
        return memory
    if torch.is_tensor(memory):
        return memory
    raise TypeError(f"Unsupported memory type: {type(memory)}")


def shortlist(
    E: torch.Tensor,
    memory: torch.Tensor | nn.Module,
    k: int,
    causal: bool = True,
    ann_index: ANNIndex | str | Path | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine shortlist supporting both table and CMM memory."""
    table = _memory_table(memory)
    n, _ = E.shape
    K = table.shape[0]
    k = min(k, K)

    loaded_index: ANNIndex | None = None
    if isinstance(ann_index, (str, Path)):
        loaded_index = load_ann_index(ann_index, table)
    elif isinstance(ann_index, ANNIndex):
        loaded_index = ann_index

    if loaded_index is not None:
        indices, _ = loaded_index.search(E, k)
        Kset = indices.long()
    else:
        En = E / (E.norm(dim=-1, keepdim=True) + 1e-9)
        Mn = table / (table.norm(dim=-1, keepdim=True) + 1e-9)
        scores = En @ Mn.t()
        topk = torch.topk(scores, k=k, dim=-1)
        Kset = topk.indices.long()

    mask = torch.ones_like(Kset, dtype=torch.bool, device=Kset.device)
    if causal and hasattr(memory, "positions_of_indices"):
        pat_pos = memory.positions_of_indices(Kset)  # type: ignore[attr-defined]
        tok_pos = torch.arange(n, device=E.device).unsqueeze(-1).expand_as(pat_pos)
        mask = pat_pos <= tok_pos
    return Kset, mask


__all__ = ["shortlist"]
