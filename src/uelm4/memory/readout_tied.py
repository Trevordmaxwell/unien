from __future__ import annotations

import torch
import torch.nn as nn


class TiedReadout(nn.Module):
    """Tied readout that mirrors a slice of the memory table."""

    def __init__(self, weight: torch.Tensor, bias: bool = True, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.register_buffer("weight", weight.clone())
        self.bias = nn.Parameter(torch.zeros(weight.shape[0])) if bias else None
        self._source: torch.Tensor | None = None

    def tie_to(self, source: torch.Tensor) -> None:
        self._source = source

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        weight = self._source if self._source is not None else self.weight
        logits = self.scale * (Y @ weight.t())
        if self.bias is not None:
            logits = logits + self.bias
        return logits


__all__ = ["TiedReadout"]
