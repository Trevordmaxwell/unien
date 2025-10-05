from __future__ import annotations

import math

import torch


def perplexity(loss: torch.Tensor) -> float:
    return float(math.exp(loss.detach().cpu().item()))


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs.clamp_min(1e-9)), dim=-1)


__all__ = ["perplexity", "entropy_from_logits"]
