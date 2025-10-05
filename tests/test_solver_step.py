import torch

from uelm4.core.types import FullCfg, ModelCfg, MemoryCfg, SolverCfg, FieldCfg
from uelm4.model.uelm4_model import UELM4


def test_forward_shapes_no_nan():
    cfg = FullCfg(
        model=ModelCfg(d=64, vocab_size=500, band=8),
        memory=MemoryCfg(K=1024, shortlist_k=16),
        solver=SolverCfg(T_train=2, rho=1.0, early_exit_tol=1e-3),
        field=FieldCfg(spectral_norm=False),
    )
    model = UELM4(cfg).eval()
    tokens = torch.randint(0, cfg.model.vocab_size, (64,))
    logits = model(tokens)
    assert logits.shape == (tokens.shape[0], cfg.model.vocab_size)
    assert torch.isfinite(logits).all()
