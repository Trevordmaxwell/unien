from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
import torch.nn.functional as F


def _align(source: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    n, d = target_shape
    if source.shape == target_shape:
        return source
    if source.shape[1] != d:
        raise ValueError("Feature dimension mismatch for cache alignment")
    if source.shape[0] >= n:
        return source[:n]
    pad = n - source.shape[0]
    padding = torch.zeros(pad, d, device=source.device, dtype=source.dtype)
    return torch.cat([source, padding], dim=0)


def _advect_identity(prev: torch.Tensor, target_shape: torch.Size, decay: float) -> torch.Tensor:
    return _align(prev, target_shape)


def _advect_shift(prev: torch.Tensor, target_shape: torch.Size, decay: float) -> torch.Tensor:
    aligned = _align(prev, target_shape)
    shifted = torch.roll(aligned, shifts=1, dims=0)
    shifted[0] = 0
    return shifted


def _advect_ema(prev: torch.Tensor, target_shape: torch.Size, decay: float) -> torch.Tensor:
    aligned = _align(prev, target_shape)
    if decay <= 0:
        return aligned
    ema = torch.zeros_like(aligned)
    ema[0] = aligned[0]
    alpha = decay
    for i in range(1, aligned.shape[0]):
        ema[i] = alpha * ema[i - 1] + (1 - alpha) * aligned[i]
    return ema


def _advect_conv(prev: torch.Tensor, target_shape: torch.Size, decay: float) -> torch.Tensor:
    aligned = _align(prev, target_shape)
    alpha = torch.clamp(torch.tensor(decay, device=aligned.device, dtype=aligned.dtype), 0.0, 1.0)
    kernel = torch.stack([alpha, 1 - alpha]).reshape(1, 1, 2)
    num_channels = aligned.shape[1]
    weight = kernel.expand(num_channels, 1, 2)
    signal = aligned.transpose(0, 1).unsqueeze(0)
    padded = F.pad(signal, (1, 0))
    filtered = F.conv1d(padded, weight, groups=num_channels)
    return filtered.squeeze(0).transpose(0, 1)


_ADVECTOR_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Size, float], torch.Tensor]] = {
    "identity": _advect_identity,
    "shift": _advect_shift,
    "ema": _advect_ema,
    "conv": _advect_conv,
}


@dataclass
class CacheState:
    """Holds previous solver output for cache-as-constraint."""

    Y_prev: torch.Tensor

    def advect(self, template: torch.Tensor, method: str, decay: float) -> torch.Tensor:
        advector = _ADVECTOR_REGISTRY.get(method, _advect_identity)
        return advector(self.Y_prev, template.shape, decay)

    def update(self, Y: torch.Tensor) -> "CacheState":
        return CacheState(Y.detach())

    @staticmethod
    def from_tensor(Y: torch.Tensor) -> "CacheState":
        return CacheState(Y.detach())


def apply_cache_penalty(
    Y: torch.Tensor,
    cache: CacheState | None,
    kappa: float,
    method: str = "identity",
    decay: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache is None or kappa <= 0.0:
        return Y, torch.zeros((), device=Y.device, dtype=Y.dtype)
    target = cache.advect(Y, method=method, decay=decay)
    penalty = Y - target
    energy = kappa * penalty.pow(2).mean()
    Y_new = Y - kappa * penalty
    return Y_new, energy


__all__ = ["CacheState", "apply_cache_penalty"]
