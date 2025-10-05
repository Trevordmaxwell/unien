from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - faiss optional
    faiss = None


@dataclass
class ANNIndex:
    """Wrapper around FAISS (if available) or normalized table fallback."""

    table: torch.Tensor
    index: Optional[object] = None
    normalize: bool = True

    def search(self, queries: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.index is not None and faiss is not None:
            q = queries.detach().cpu().numpy().astype("float32")
            scores, idx = self.index.search(q, k)
            return torch.from_numpy(idx).to(queries.device), torch.from_numpy(scores).to(queries.device)
        table = self.table.to(queries.device)
        if self.normalize:
            table = table / (table.norm(dim=-1, keepdim=True) + 1e-9)
            queries = queries / (queries.norm(dim=-1, keepdim=True) + 1e-9)
        scores = queries @ table.t()
        return torch.topk(scores, k=min(k, table.shape[0]), dim=-1).indices, scores


def build_ann_index(table: torch.Tensor, nlist: int = 100, nprobe: int = 8) -> ANNIndex:
    if faiss is None:
        return ANNIndex(table=table.detach(), index=None, normalize=True)
    d = table.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    feats = table.detach().cpu().numpy().astype("float32")
    index.train(feats)
    index.add(feats)
    index.nprobe = nprobe
    return ANNIndex(table=table.detach(), index=index, normalize=False)


def load_ann_index(meta_path: str | Path, table: torch.Tensor) -> ANNIndex:
    meta_path = Path(meta_path)
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    index_type = meta.get("type", "cosine")
    if index_type == "faiss":
        if faiss is None:
            raise RuntimeError("FAISS is not available but index metadata requires it")
        idx_path = meta.get("index_path")
        if idx_path is None:
            raise ValueError("FAISS metadata missing 'index_path'")
        faiss_index = faiss.read_index(str(meta_path.parent / idx_path))
        if "nprobe" in meta:
            faiss_index.nprobe = meta["nprobe"]
        return ANNIndex(table=table.detach(), index=faiss_index, normalize=False)
    normalize = bool(meta.get("normalize", True))
    return ANNIndex(table=table.detach(), index=None, normalize=normalize)


__all__ = ["ANNIndex", "build_ann_index", "load_ann_index"]
