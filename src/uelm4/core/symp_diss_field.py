import torch
import torch.nn as nn

from .banded_ops import BandedSpec, CausalBandedConv1d
from .control_path import Path


class BandedField(nn.Module):
    """
    Simplified causal banded operator producing a residual on Y.
    Combines a symplectic (skew) and dissipative component implemented via
    causal convolutions.
    """

    def __init__(self, d: int, band: int, spectral_norm: bool = False):
        super().__init__()
        spec = BandedSpec(channels=d, band=band, bias=False, spectral_norm=spectral_norm)
        self.symplectic = CausalBandedConv1d(spec)
        self.dissipative = CausalBandedConv1d(spec)

    def forward(self, Y: torch.Tensor, t_feat: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        S = self.symplectic(Y)
        D = self.dissipative(Y)
        return S, D


def cde_split_step(Y: torch.Tensor, X: Path, field: BandedField, step: float = 1.0) -> torch.Tensor:
    """
    One simplified Strang-like step:
      Y <- Y + 0.5*S(Y)
      Y <- Y - D(Y)          (dissipative correction)
      Y <- Y + 0.5*S(Y)
    """
    S1, _ = field(Y, None)
    Y = Y + 0.5 * step * S1
    _, D2 = field(Y, None)
    Y = Y - step * D2
    S3, _ = field(Y, None)
    Y = Y + 0.5 * step * S3
    return Y
