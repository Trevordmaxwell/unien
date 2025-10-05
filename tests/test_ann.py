import json
import torch

from uelm4.memory.ann import build_ann_index, load_ann_index


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
