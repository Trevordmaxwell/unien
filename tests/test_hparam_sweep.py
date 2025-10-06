from pathlib import Path

import torch

from uelm4.train.hparam_sweep import run_sweep


def test_run_sweep(tmp_path):
    corpus = tmp_path / "sample.txt"
    corpus.write_text("alpha beta gamma delta")
    sweeps = [
        {"model": {"d": 64, "vocab_size": 128}, "solver": {"T_train": 1}},
        {"model": {"d": 64, "vocab_size": 128}, "solver": {"T_train": 2}},
    ]
    results = run_sweep("small", corpus, sweeps, epochs=1, device=torch.device("cpu"))
    assert len(results) == 2
    for res in results:
        metrics = res.metrics
        assert "loss" in metrics and "energy" in metrics
        assert "perplexity" in metrics and "iters_per_token" in metrics
        assert metrics["loss"] >= 0
