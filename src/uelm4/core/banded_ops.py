from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BandedSpec:
    channels: int
    band: int
    bias: bool = False
    spectral_norm: bool = False


class CausalBandedConv1d(nn.Module):
    """Thin wrapper over :class:`nn.Conv1d` that enforces causal padding."""

    def __init__(self, spec: BandedSpec):
        super().__init__()
        self.spec = spec
        self.band = max(1, int(spec.band))
        conv = nn.Conv1d(spec.channels, spec.channels, kernel_size=self.band, bias=spec.bias)
        if spec.spectral_norm:
            conv = nn.utils.parametrizations.spectral_norm(conv)
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_was_seq = False
        if x.ndim == 2:
            x = x.transpose(0, 1).unsqueeze(0)
            x_was_seq = True
        pad = (self.band - 1, 0)
        out = self.conv(F.pad(x, pad))
        if x_was_seq:
            out = out.squeeze(0).transpose(0, 1)
        return out


def apply_causal_kernels(x: torch.Tensor, kernels: Iterable[CausalBandedConv1d]) -> list[torch.Tensor]:
    """Apply a list of causal banded convolutions to *x* returning the outputs."""
    return [kernel(x) for kernel in kernels]


__all__ = ["BandedSpec", "CausalBandedConv1d", "apply_causal_kernels"]
