import torch

from uelm4.core.energy import compute_energy_terms, total_energy
from uelm4.core.solver_pdhg import SolverState


def test_energy_terms_non_negative():
    n, k, d = 4, 3, 2
    P = torch.full((n, k), 1.0 / k)
    Y = torch.randn(n, d)
    Lam = torch.zeros(n, d)
    Kset = torch.zeros(n, k, dtype=torch.long)
    M = torch.randn(8, d)
    state = SolverState(P=P, Y=Y, Lam=Lam, Kset=Kset, energy=0.0)
    terms = compute_energy_terms(state, M, rho=1.0)
    for value in terms.values():
        scalar = float(value.detach())
        assert scalar >= 0.0
    energy = float(total_energy(state, M, rho=1.0).detach())
    assert energy >= 0.0
