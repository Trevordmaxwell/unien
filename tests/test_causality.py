import torch

from uelm4.core.symp_diss_field import BandedField, cde_split_step
from uelm4.core.control_path import Path


def test_causal_banded_no_future_leak():
    n, d, band = 64, 32, 8
    Y = torch.zeros(n, d)
    Y[-1] = torch.randn(d)
    field = BandedField(d, band, spectral_norm=False)
    with torch.no_grad():
        field.conv_s.weight[..., -1] += 1e-3
        field.conv_d.weight[..., -1] += 1e-3
    X = Path(times=torch.linspace(0, 1, n), knots=torch.zeros(n, d))
    Y_out = cde_split_step(Y, X, field)
    assert Y_out[: n - band - 1].abs().max() < 1e-3 + 1e-6
