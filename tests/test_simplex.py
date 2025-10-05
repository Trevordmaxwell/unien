import torch

from uelm4.core.kl_prox import kl_masked_softmax


def test_kl_prox_simplex():
    n, k = 8, 5
    P = torch.full((n, k), 1.0 / k)
    scores = torch.randn(n, k)
    out = kl_masked_softmax(P, scores)
    assert torch.allclose(out.sum(dim=-1), torch.ones(n), atol=1e-6)
    assert torch.all(out >= 0)
