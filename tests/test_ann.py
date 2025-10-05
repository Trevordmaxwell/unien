import json
import torch
import pytest

from uelm4.memory.ann import build_ann_index, load_ann_index

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def test_ann_index_fallback_matches_topk():
    table = torch.randn(32, 8)
    queries = torch.randn(4, 8)
    index = build_ann_index(table)
    idx, _ = index.search(queries, k=5)
    assert idx.shape == (4, 5)
    assert (idx >= 0).all()


def test_load_ann_index_cosine(tmp_path):
    table = torch.randn(16, 4)
    meta = {"type": "cosine", "normalize": True}
    meta_path = tmp_path / "index.json"
    meta_path.write_text(json.dumps(meta))
    index = load_ann_index(meta_path, table)
    idx, _ = index.search(torch.randn(3, 4), k=3)
    assert idx.shape == (3, 3)


@pytest.mark.skipif(faiss is None, reason="FAISS not installed")
def test_load_ann_index_faiss(tmp_path):
    d = 8
    table = torch.randn(64, d)
    index = build_ann_index(table, nlist=16, nprobe=4)
    ann_path = tmp_path / "ann_index.faiss"
    meta_path = tmp_path / "ann_index.json"
    faiss.write_index(index.index, str(ann_path))  # type: ignore[attr-defined]
    meta_path.write_text(json.dumps({
        "type": "faiss",
        "index_path": ann_path.name,
        "nlist": 16,
        "nprobe": 4,
    }))
    loaded = load_ann_index(meta_path, table)
    queries = torch.randn(2, d)
    idx, _ = loaded.search(queries, k=5)
    assert idx.shape == (2, 5)
