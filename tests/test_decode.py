import torch

from uelm4.core.types import FieldCfg, FullCfg, MemoryCfg, ModelCfg, SolverCfg
from uelm4.model.decode import greedy_decode
from uelm4.model.uelm4_model import UELM4


def test_greedy_decode_extends_sequence():
    cfg = FullCfg(
        model=ModelCfg(d=32, vocab_size=128, band=4),
        memory=MemoryCfg(K=256, shortlist_k=8, type="table", K0=64),
        solver=SolverCfg(T_train=1, T_infer=1, rho=1.0, early_exit_tol=1e-3),
        field=FieldCfg(spectral_norm=False),
    )
    model = UELM4(cfg).eval()
    prompt = torch.randint(0, cfg.model.vocab_size, (5,))
    generated = greedy_decode(model, prompt, max_new_tokens=2)
    assert generated.shape[0] == prompt.shape[0] + 2
