"""Configuration utilities for UELM-4."""

from pathlib import Path
from dataclasses import asdict
from typing import Any, Mapping

import yaml

from ..core.types import FullCfg, ModelCfg, MemoryCfg, SolverCfg, FieldCfg, CACCfg

CONFIG_ROOT = Path(__file__).resolve().parent

__all__ = [
    "CONFIG_ROOT",
    "load_config",
    "FullCfg",
    "ModelCfg",
    "MemoryCfg",
    "SolverCfg",
    "FieldCfg",
    "CACCfg",
]


def _dict_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _dict_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(name: str | Path, overrides: Mapping[str, Any] | None = None) -> FullCfg:
    """Load a YAML config into a :class:`FullCfg` dataclass."""
    path = Path(name)
    if not path.suffix:
        path = CONFIG_ROOT / f"{path}.yaml"
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if overrides:
        data = _dict_merge(data, overrides)
    model = ModelCfg(**data.get("model", {}))
    memory = MemoryCfg(**data.get("memory", {}))
    solver = SolverCfg(**data.get("solver", {}))
    field = FieldCfg(**data.get("field", {}))
    cac = CACCfg(**data.get("cac", {}))
    return FullCfg(model=model, memory=memory, solver=solver, field=field, cac=cac)


if __name__ == "__main__":
    cfg = load_config("base16k")
    print(asdict(cfg))
