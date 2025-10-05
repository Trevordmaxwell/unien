import torch

from uelm4.core.wmf_prox import wmf_prox_step


def test_wmf_keeps_simplex():
    n, k = 2, 4
    P = torch.full((n, k), 1.0 / k)
    scores = torch.randn(n, k)
    cost = torch.rand(n, k, k)
    mask = torch.ones(n, k, dtype=torch.bool)
    out = wmf_prox_step(
        P,
        scores,
        cost,
        tau=0.1,
        lam_kl=1.0,
        num_iters=4,
        eps=1e-3,
        cost_scale=0.7,
        mask=mask,
    )
    assert torch.allclose(out.sum(dim=-1), torch.ones(n), atol=1e-5)
    assert (out >= 0).all()


def test_wmf_respects_mask():
    n, k = 1, 3
    P = torch.tensor([[0.4, 0.4, 0.2]])
    scores = torch.tensor([[1.0, 0.5, -2.0]])
    cost = torch.ones(n, k, k)
    mask = torch.tensor([[True, True, False]])
    out = wmf_prox_step(P, scores, cost, tau=0.2, lam_kl=1.0, mask=mask)
    assert torch.all(out[..., -1] <= 1e-3)
    assert torch.allclose(out.sum(dim=-1), torch.ones(n), atol=1e-5)
