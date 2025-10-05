from dataclasses import dataclass, field as dataclass_field


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
    wmf_iters: int = 3
    wmf_eps: float = 1.0e-3
    wmf_cost_scale: float = 1.0


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
    meta_path: str | None = None


@dataclass
class FieldCfg:
    spectral_norm: bool = False


@dataclass
class CACCfg:
    kappa: float = 0.0
    advect: str = "identity"
    decay: float = 0.0


@dataclass
class FullCfg:
    model: ModelCfg = dataclass_field(default_factory=ModelCfg)
    memory: MemoryCfg = dataclass_field(default_factory=MemoryCfg)
    solver: SolverCfg = dataclass_field(default_factory=SolverCfg)
    field: FieldCfg = dataclass_field(default_factory=FieldCfg)
    cac: CACCfg = dataclass_field(default_factory=CACCfg)
