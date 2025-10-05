from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _memory_table(memory: torch.Tensor | nn.Module) -> torch.Tensor:
    if hasattr(memory, "landmarks_view"):
        return getattr(memory, "landmarks_view")()
    if isinstance(memory, nn.Parameter):
        return memory
    if torch.is_tensor(memory):
        return memory
    raise TypeError(f"Unsupported memory type: {type(memory)}")


def shortlist(E: torch.Tensor, memory: torch.Tensor | nn.Module, k: int, causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine shortlist supporting both table and CMM memory."""
    table = _memory_table(memory)
    n, _ = E.shape
    K = table.shape[0]
    k = min(k, K)
    En = E / (E.norm(dim=-1, keepdim=True) + 1e-9)
    Mn = table / (table.norm(dim=-1, keepdim=True) + 1e-9)
    scores = En @ Mn.t()
    topk = torch.topk(scores, k=k, dim=-1)
    Kset = topk.indices.long()
    mask = torch.ones_like(Kset, dtype=torch.bool, device=Kset.device)
    if causal:
        positions = torch.arange(n, device=E.device).unsqueeze(-1)
        causal_mask = Kset <= positions.max()
        mask = mask & causal_mask
    return Kset, mask


__all__ = ["shortlist"]
