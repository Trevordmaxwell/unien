from dataclasses import dataclass, field


@dataclass
class SolverCfg:
    T_train: int = 2
    T_infer: int = 1
    rho: float = 1.0
    beta_start: float = 0.5
    beta_end: float = 1.5
    tau_start: float = 0.0
    tau_end: float = 0.0
    use_wmf: bool = False
    early_exit_tol: float = 1.0e-3


@dataclass
class ModelCfg:
    d: int = 256
    vocab_size: int = 32000
    band: int = 16
    tied_readout: bool = True


@dataclass
class MemoryCfg:
    K: int = 4096
    shortlist_k: int = 32
    type: str = "table"
    K0: int = 1024


@dataclass
class FieldCfg:
    spectral_norm: bool = False


@dataclass
class FullCfg:
    model: ModelCfg = field(default_factory=ModelCfg)
    memory: MemoryCfg = field(default_factory=MemoryCfg)
    solver: SolverCfg = field(default_factory=SolverCfg)
    field: FieldCfg = field(default_factory=FieldCfg)
