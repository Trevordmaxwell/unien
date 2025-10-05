import torch

from uelm4.core.cac import CacheState, apply_cache_penalty


def test_cache_penalty_moves_towards_cache_identity():
    Y = torch.zeros(4, 3)
    cache = CacheState.from_tensor(torch.ones(4, 3))
    Y_new, energy = apply_cache_penalty(Y, cache, kappa=1.0, method="identity")
    assert torch.allclose(Y_new, torch.ones_like(Y))
    assert energy >= 0.0


def test_cache_alignment_padding():
    cache = CacheState.from_tensor(torch.ones(2, 3))
    target = torch.zeros(4, 3)
    aligned = cache.advect(target, method="identity", decay=0.0)
    assert aligned.shape == target.shape
    assert torch.allclose(aligned[:2], torch.ones(2, 3))


def test_cache_shift_advect():
    cache = CacheState.from_tensor(torch.arange(6.0).reshape(3, 2))
    target = torch.zeros(4, 2)
    shifted = cache.advect(target, method="shift", decay=0.0)
    assert torch.allclose(shifted[1], torch.tensor([0.0, 1.0]))


def test_cache_ema_advect():
    cache = CacheState.from_tensor(torch.ones(3, 2))
    target = torch.zeros(3, 2)
    ema = cache.advect(target, method="ema", decay=0.5)
    assert torch.all(ema >= 0)
    assert torch.all(ema <= 1)


def test_cache_conv_advect():
    cache = CacheState.from_tensor(torch.arange(6.0).reshape(3, 2))
    target = torch.zeros(3, 2)
    conv = cache.advect(target, method="conv", decay=0.3)
    assert conv.shape == target.shape
    assert torch.isfinite(conv).all()
