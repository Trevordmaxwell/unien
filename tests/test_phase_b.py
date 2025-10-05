import torch

from uelm4.config import load_config
from uelm4.memory.bootstrap import build_phase_b_assets
from uelm4.memory.ann import load_ann_index
from uelm4.model.uelm4_model import UELM4


def test_phase_b_bootstrap_assets(tmp_path):
    d = 12
    table = torch.randn(256, d)
    lm_meta, ann_meta = build_phase_b_assets(table, tmp_path, num_landmarks=64, ann_nlist=32, ann_nprobe=4)

    cfg = load_config(
        "small",
        {
            "model": {"d": d, "vocab_size": 64},
            "memory": {"type": "cmm", "K0": 64, "meta_path": str(lm_meta)},
            "solver": {"use_wmf": True, "wmf_iters": 2, "wmf_eps": 1e-3},
        },
    )
    model = UELM4(cfg).eval()
    tokens = torch.randint(0, cfg.model.vocab_size, (24,))
    logits = model(tokens, ann_index=str(ann_meta))
    assert logits.shape == (tokens.shape[0], cfg.model.vocab_size)
    assert torch.isfinite(logits).all()

    table_for_ann = model.memory.landmarks_view() if hasattr(model.memory, "landmarks_view") else model.memory
    index = load_ann_index(ann_meta, table_for_ann)
    idx, _ = index.search(torch.randn(4, d), k=8)
    assert idx.shape == (4, 8)
