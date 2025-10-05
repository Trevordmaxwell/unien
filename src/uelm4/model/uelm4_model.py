import torch
import torch.nn as nn

from ..core.types import FullCfg
from ..model.embeddings import Embeddings
from ..core.control_path import make_control_path
from ..core.solver_pdhg import MirrorPDHG, SolverState, M_T_P
from ..core.symp_diss_field import BandedField
from ..memory.shortlist import shortlist
from ..memory.readout_tied import TiedReadout


class UELM4(nn.Module):
    def __init__(self, cfg: FullCfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg.model.vocab_size, cfg.model.d)
        self.field = BandedField(cfg.model.d, cfg.model.band, spectral_norm=cfg.field.spectral_norm)
        # Phase-A: table memory
        self.M = nn.Parameter(torch.randn(cfg.memory.K, cfg.model.d) * 0.02)
        # Tied readout uses first vocab_size rows (simple tie for MVP)
        self.readout = TiedReadout(self.M[: cfg.model.vocab_size].detach())
        self.solver = MirrorPDHG(rho=cfg.solver.rho, field=self.field)

    def forward(self, tokens: torch.Tensor, T: int | None = None) -> torch.Tensor:
        cfg = self.cfg
        n = tokens.shape[0]
        E = self.embed(tokens)                       # (n,d)
        X = make_control_path(E)
        Kset, _ = shortlist(E, self.M, cfg.memory.shortlist_k)  # (n,k)
        # Init P uniform on shortlist
        P0 = torch.full((n, Kset.shape[1]), 1.0 / Kset.shape[1], device=tokens.device)
        Y0 = M_T_P(self.M, Kset, P0)
        Lam0 = torch.zeros_like(Y0)
        st = SolverState(P=P0, Y=Y0, Lam=Lam0, Kset=Kset, energy=float("inf"))

        iters = T if T is not None else cfg.solver.T_train
        for _ in range(iters):
            st = self.solver.step(st, X, self.M)
            if st.energy <= cfg.solver.early_exit_tol:
                break

        logits = self.readout(st.Y)                  # (n, vocab)
        return logits
