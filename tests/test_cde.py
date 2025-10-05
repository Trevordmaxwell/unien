import torch

from uelm4.core.symp_diss_field import BandedField, cde_split_step
from uelm4.core.control_path import Path


def test_banded_field_shapes():
    n, d, band = 16, 8, 4
    Y = torch.randn(n, d)
    field = BandedField(d, band, spectral_norm=False)
    path = Path(times=torch.linspace(0, 1, n), knots=torch.zeros(n, d))
    out = cde_split_step(Y, path, field)
    assert out.shape == Y.shape
