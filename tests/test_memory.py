import torch

from uelm4.core.types import FieldCfg, FullCfg, MemoryCfg, ModelCfg, SolverCfg
from uelm4.memory.cmm import CMMemory
from uelm4.memory.landmark_io import save_landmarks
from uelm4.memory.shortlist import shortlist
from uelm4.model.uelm4_model import UELM4


def test_cmm_sample_shapes():
    d, num_landmarks = 16, 32
    memory = CMMemory(d, num_landmarks)
    queries = torch.randn(4, d)
    atoms = memory.sample_atoms(queries, num=6)
    assert atoms.shape == (6, d)


def test_shortlist_with_cmm_memory():
    d = 16
    memory = CMMemory(d, 32)
    E = torch.randn(5, d)
    Kset, mask = shortlist(E, memory, k=4)
    assert Kset.shape == (5, 4)
    assert mask.shape == (5, 4)


def test_cmm_meta_path_load(tmp_path):
    d = 16
    landmarks = torch.randn(12, d)
    _, meta_path = save_landmarks(landmarks, tmp_path / "landmarks")
    cfg = FullCfg(
        model=ModelCfg(d=d, vocab_size=128, band=4),
        memory=MemoryCfg(K=256, shortlist_k=4, type="cmm", K0=landmarks.shape[0], meta_path=str(meta_path)),
        solver=SolverCfg(T_train=1, T_infer=1, rho=1.0, early_exit_tol=1e-4),
        field=FieldCfg(spectral_norm=False),
    )
    model = UELM4(cfg).eval()
    actual = model.memory.landmarks_view().detach().cpu()
    assert torch.allclose(actual, landmarks)

