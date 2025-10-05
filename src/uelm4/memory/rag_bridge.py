from __future__ import annotations

from typing import Iterable, Tuple

import torch


def inject_rag_atoms(memory: torch.Tensor, rag_vectors: Iterable[torch.Tensor], max_atoms: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Append external retrieval vectors to the memory table."""
    rag = torch.stack(list(rag_vectors)) if not isinstance(rag_vectors, torch.Tensor) else rag_vectors
    if max_atoms is not None and rag.shape[0] > max_atoms:
        rag = rag[:max_atoms]
    augmented = torch.cat([memory, rag.to(memory.device)], dim=0)
    indices = torch.arange(augmented.shape[0], device=augmented.device)
    return augmented, indices


__all__ = ["inject_rag_atoms"]
