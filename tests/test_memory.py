import torch

from uelm4.memory.cmm import CMMemory
from uelm4.memory.shortlist import shortlist


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
