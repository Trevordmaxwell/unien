import torch

from uelm4.core.wmf_prox import wmf_prox_step


def test_wmf_keeps_simplex():
    n, k = 2, 4
    P = torch.full((n, k), 1.0 / k)
    scores = torch.randn(n, k)
    cost = torch.rand(n, k, k)
    out = wmf_prox_step(P, scores, cost, tau=0.1, lam_kl=1.0)
    assert torch.allclose(out.sum(dim=-1), torch.ones(n), atol=1e-5)
    assert (out >= 0).all()
