from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from ..core.cac import CacheState
from ..core.control_path import make_control_path
from ..core.solver_pdhg import MirrorPDHG, SolverState
from ..core.symp_diss_field import BandedField
from ..core.types import FullCfg
from ..memory.ann import ANNIndex, load_ann_index
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
        self.solver = MirrorPDHG(cfg.solver, self.field, controller, cac_cfg=cfg.cac)
        if cfg.model.tied_readout:
            lex_weight = self._lexicon_weight()
            self.readout = TiedReadout(lex_weight.detach())
            self.readout.tie_to(lex_weight)
        else:
            self.readout = nn.Linear(cfg.model.d, cfg.model.vocab_size)

    def _init_memory(self) -> nn.Module | torch.Tensor:
        cfg = self.cfg
        if cfg.memory.type == "cmm":
            return CMMemory(cfg.model.d, cfg.memory.K0, meta_path=cfg.memory.meta_path)
        return nn.Parameter(torch.randn(cfg.memory.K, cfg.model.d) * 0.02)

    def _lexicon_weight(self) -> torch.Tensor:
        if isinstance(self.memory, CMMemory):
            table = self.memory.landmarks_view()
        else:
            table = self.memory
        return table[: self.cfg.model.vocab_size]

    def _retie_readout_if_needed(self) -> None:
        if isinstance(self.readout, TiedReadout):
            self.readout.tie_to(self._lexicon_weight())

    def forward(
        self,
        tokens: torch.Tensor,
        T: int | None = None,
        cache: CacheState | None = None,
        ann_index: ANNIndex | str | Path | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SolverState, CacheState]:
        cfg = self.cfg
        E = self.embed(tokens)
        X = make_control_path(E)
        memory_table = self.memory.landmarks_view() if isinstance(self.memory, CMMemory) else self.memory
        if isinstance(ann_index, (str, Path)):
            ann_index = load_ann_index(ann_index, memory_table)
        if cfg.model.tied_readout:
            self._retie_readout_if_needed()
        Kset, _ = shortlist(E, self.memory, cfg.memory.shortlist_k, ann_index=ann_index)

        P0, Y0 = self.scout(E, memory_table, Kset)
        Lam0 = torch.zeros_like(Y0)
        st = SolverState(P=P0, Y=Y0, Lam=Lam0, Kset=Kset, energy=float("inf"))

        cache_state = cache
        iters = T if T is not None else cfg.solver.T_train
        self.solver.reset()
        prev_energy = float("inf")
        for _ in range(iters):
            st, cache_state = self.solver.step(st, X, memory_table, cache=cache_state)
            if prev_energy != float("inf"):
                delta = abs(prev_energy - st.energy)
                denom = max(abs(prev_energy), 1e-6)
                rel_drop = delta / denom
                if rel_drop <= cfg.solver.early_exit_tol:
                    break
            prev_energy = st.energy

        logits = self.readout(st.Y)
        if return_state:
            cache_state = cache_state or CacheState.from_tensor(st.Y)
            return logits, st, cache_state
        return logits
