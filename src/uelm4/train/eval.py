from __future__ import annotations

from typing import Iterable

import torch

from ..model.uelm4_model import UELM4
from ..train.metrics import perplexity


def evaluate_model(model: UELM4, batches: Iterable[torch.Tensor], targets: Iterable[torch.Tensor]) -> dict[str, float]:
    losses = []
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for tokens, labels in zip(batches, targets):
            logits = model(tokens)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            losses.append(loss)
    if not losses:
        return {"loss": 0.0, "ppl": 0.0}
    loss_tensor = torch.stack(losses).mean()
    return {"loss": float(loss_tensor), "ppl": perplexity(loss_tensor)}


__all__ = ["evaluate_model"]
