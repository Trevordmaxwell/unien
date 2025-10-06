from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import time

import torch

from ..config import load_config
from ..data import LoaderConfig, SimpleTokenizer, build_dataloader
from ..model.decode import greedy_decode
from ..model.uelm4_model import UELM4
from .train import train_epoch


@dataclass
class SweepResult:
    config_overrides: Dict[str, Any]
    metrics: Dict[str, float]


def _flatten_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def run_sweep(
    config_name: str,
    corpus_path: Path,
    sweeps: Iterable[Dict[str, Any]],
    epochs: int = 1,
    device: torch.device | None = None,
) -> List[SweepResult]:
    text = corpus_path.read_text(encoding="utf-8")
    lines = _flatten_lines(text)
    if not lines:
        raise ValueError("Corpus must contain non-empty lines")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = set(" ".join(lines).split())
    tokenizer = SimpleTokenizer(vocab=vocab)

    results: List[SweepResult] = []
    for overrides in sweeps:
        cfg = load_config(config_name, overrides)
        model = UELM4(cfg).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4)
        loader_cfg = LoaderConfig(max_length=cfg.solver.T_train * max(cfg.memory.shortlist_k, 4), batch_size=2, shuffle=True)
        dataloader = build_dataloader(lines, tokenizer, loader_cfg)
        metrics = {}
        for _ in range(max(1, epochs)):
            metrics = train_epoch(model, optimiser, dataloader, device)

        prompt_len = min(128, cfg.model.vocab_size)
        prompt = torch.randint(0, cfg.model.vocab_size, (prompt_len,), device=device)
        forward_start = time.perf_counter()
        with torch.no_grad():
            _, state, _ = model(prompt, return_state=True)
        forward_ms = (time.perf_counter() - forward_start) * 1000
        metrics["forward_ms"] = forward_ms
        metrics["forward_iters_per_token"] = state.iters / max(int(prompt.numel()), 1)

        decode_start = time.perf_counter()
        with torch.no_grad():
            greedy_decode(model, prompt.clone(), max_new_tokens=32)
        decode_ms = (time.perf_counter() - decode_start) * 1000
        metrics["decode_ms_per_token"] = decode_ms / 32

        results.append(SweepResult(config_overrides=overrides, metrics=metrics))
    return results


__all__ = ["run_sweep", "SweepResult"]
