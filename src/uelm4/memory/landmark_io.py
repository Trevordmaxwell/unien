from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch


def save_landmarks(tensor: torch.Tensor, base_path: str | Path) -> Tuple[Path, Path]:
    base_path = Path(base_path)
    tensor_path = base_path.with_suffix(".pt") if base_path.suffix != ".pt" else base_path
    meta_path = tensor_path.with_suffix(".json")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, tensor_path)
    meta = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "path": tensor_path.name,
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return tensor_path, meta_path


def load_landmarks(meta_path: str | Path, root: str | Path | None = None) -> torch.Tensor:
    meta_path = Path(meta_path)
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    tensor_rel = meta.get("path")
    tensor_path = meta_path.parent / tensor_rel if tensor_rel else meta_path.with_suffix(".pt")
    if root is not None:
        tensor_path = Path(root) / tensor_path.name
    tensor = torch.load(tensor_path, map_location="cpu")
    expected_shape = tuple(meta.get("shape", tensor.shape))
    if tensor.shape != expected_shape:
        raise ValueError(f"Landmark shape mismatch: expected {expected_shape}, got {tensor.shape}")
    return tensor


__all__ = ["save_landmarks", "load_landmarks"]
