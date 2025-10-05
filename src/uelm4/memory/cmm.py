from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GeneratorCfg:
    hidden_dim: int = 128
    depth: int = 2


class LandmarkGenerator(nn.Module):
    """Toy generator that refines landmarks given seed queries."""

    def __init__(self, d: int, cfg: GeneratorCfg):
        super().__init__()
        layers: list[nn.Module] = []
        last = d
        for _ in range(cfg.depth - 1):
            layers.extend([nn.Linear(last, cfg.hidden_dim), nn.GELU()])
            last = cfg.hidden_dim
        layers.append(nn.Linear(last, d))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CMMemory(nn.Module):
    """Continuous Memory Measure represented by learnable landmarks."""

    def __init__(self, d: int, num_landmarks: int, generator_cfg: GeneratorCfg | None = None):
        super().__init__()
        self.landmarks = nn.Parameter(torch.randn(num_landmarks, d) * 0.02)
        self.generator = LandmarkGenerator(d, generator_cfg or GeneratorCfg())

    def forward(self) -> torch.Tensor:
        return self.landmarks

    def landmarks_view(self) -> torch.Tensor:
        return self.landmarks

    def sample_atoms(self, queries: torch.Tensor, num: int) -> torch.Tensor:
        reps = self.generator(queries)
        weights = torch.softmax(reps @ self.landmarks.t(), dim=-1)
        atoms = torch.matmul(weights, self.landmarks)
        if num <= atoms.shape[0]:
            return atoms[:num]
        pad = atoms.new_zeros(num - atoms.shape[0], atoms.shape[1])
        return torch.cat([atoms, pad], dim=0)


__all__ = ["CMMemory", "GeneratorCfg", "LandmarkGenerator"]
