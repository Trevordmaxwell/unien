from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch

from .landmark_io import save_landmarks
from .landmarks import select_landmarks
from .ann import build_ann_index

try:  # pragma: no cover
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def build_phase_b_assets(
    table: torch.Tensor,
    out_dir: str | Path,
    *,
    num_landmarks: int = 1024,
    ann_nlist: int = 128,
    ann_nprobe: int = 8,
) -> Tuple[Path, Path]:
    """Create landmark and ANN metadata for Phase-B runs.

    Returns a tuple (landmark_meta_path, ann_meta_path).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    _, landmarks = select_landmarks(table, num_landmarks)
    _, lm_meta_path = save_landmarks(landmarks, out_path / "landmarks")

    ann_index = build_ann_index(landmarks, nlist=ann_nlist, nprobe=ann_nprobe)
    ann_meta_path = out_path / "ann_index.json"
    if ann_index.index is not None and faiss is not None:
        index_path = out_path / "ann_index.faiss"
        faiss.write_index(ann_index.index, str(index_path))  # type: ignore[attr-defined]
        ann_meta = {
            "type": "faiss",
            "index_path": index_path.name,
            "nlist": ann_nlist,
            "nprobe": ann_nprobe,
        }
    else:
        ann_meta = {"type": "cosine", "normalize": ann_index.normalize}
    ann_meta_path.write_text(json.dumps(ann_meta, indent=2))
    return lm_meta_path, ann_meta_path


__all__ = ["build_phase_b_assets"]
