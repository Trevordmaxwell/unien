from __future__ import annotations

import torch
import torch.nn as nn

from ..core.control_path import make_control_path
from ..core.solver_pdhg import MirrorPDHG, SolverState
from ..core.symp_diss_field import BandedField
from ..core.types import FullCfg
from ..memory.cmm import CMMemory
from ..memory.readout_tied import TiedReadout
from ..memory.shortlist import shortlist
from ..model.controller import MetaController
from ..model.embeddings import Embeddings
from ..model.scout import Scout


class UELM4(nn.Module):
    def __init__(self, cfg: FullCfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg.model.vocab_size, cfg.model.d)
        self.field = BandedField(cfg.model.d, cfg.model.band, spectral_norm=cfg.field.spectral_norm)
        self.memory = self._init_memory()
        self.scout = Scout(cfg.model.d)
        controller = MetaController(
            (cfg.solver.beta_start, cfg.solver.beta_end),
            (cfg.solver.tau_start, cfg.solver.tau_end),
        )
        self.solver = MirrorPDHG(cfg.solver, self.field, controller)
        if cfg.model.tied_readout:
            lex_weight = self._lexicon_weight().detach()
            self.readout = TiedReadout(lex_weight)
        else:
            self.readout = nn.Linear(cfg.model.d, cfg.model.vocab_size)

    def _init_memory(self) -> nn.Module | torch.Tensor:
        cfg = self.cfg
        if cfg.memory.type == "cmm":
            return CMMemory(cfg.model.d, cfg.memory.K0)
        return nn.Parameter(torch.randn(cfg.memory.K, cfg.model.d) * 0.02)

    def _lexicon_weight(self) -> torch.Tensor:
        if isinstance(self.memory, CMMemory):
            table = self.memory.landmarks_view()
        else:
            table = self.memory
        return table[: self.cfg.model.vocab_size]

    def forward(self, tokens: torch.Tensor, T: int | None = None) -> torch.Tensor:
        cfg = self.cfg
        E = self.embed(tokens)
        X = make_control_path(E)
        Kset, _ = shortlist(E, self.memory, cfg.memory.shortlist_k)
        if cfg.model.tied_readout:
            self.readout.tie_to(self._lexicon_weight())

        memory_table = self.memory.landmarks_view() if isinstance(self.memory, CMMemory) else self.memory
        P0, Y0 = self.scout(E, memory_table, Kset)
        Lam0 = torch.zeros_like(Y0)
        st = SolverState(P=P0, Y=Y0, Lam=Lam0, Kset=Kset, energy=float("inf"))

        iters = T if T is not None else cfg.solver.T_train
        for _ in range(iters):
            st = self.solver.step(st, X, memory_table)
            if st.energy <= cfg.solver.early_exit_tol:
                break

        logits = self.readout(st.Y)
        return logits
