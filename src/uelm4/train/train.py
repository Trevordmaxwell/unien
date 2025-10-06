from __future__ import annotations

from typing import Iterable

import math

import torch
from torch.optim import AdamW

from ..config import load_config
from ..data.dataloaders import LoaderConfig, build_dataloader
from ..data.tokenization import SimpleTokenizer
from ..model.uelm4_model import UELM4
from ..train.losses import total_loss


def _sequence_batch(batch: torch.Tensor) -> torch.Tensor:
    return batch.unsqueeze(0) if batch.ndim == 1 else batch


def train_epoch(model: UELM4, optimiser: AdamW, dataloader: Iterable[torch.Tensor], device: torch.device) -> dict[str, float]:
    model.train()
    total_loss_value = 0.0
    total_energy = 0.0
    total_iters = 0.0
    total_tokens = 0
    batches = 0
    for batch in dataloader:
        optimiser.zero_grad()
        seqs = _sequence_batch(batch)
        num_seqs = max(seqs.shape[0], 1)
        batch_loss = 0.0
        batch_energy = 0.0
        batch_iters = 0.0
        batch_tokens = 0
        for seq in seqs:
            seq = seq.to(device)
            logits, state, _ = model(seq, return_state=True)
            seq_loss = total_loss(logits, seq, state)
            seq_loss.backward()
            batch_loss += float(seq_loss.detach())
            batch_energy += float(state.energy)
            # Some solver states may not expose iteration counts
            batch_iters += float(getattr(state, "iters", 0.0))
            batch_tokens += int(seq.numel())
        optimiser.step()
        total_loss_value += batch_loss / num_seqs
        total_energy += batch_energy / num_seqs
        total_iters += batch_iters
        total_tokens += batch_tokens
        batches += 1
    denom = max(batches, 1)
    avg_loss = total_loss_value / denom
    avg_energy = total_energy / denom
    perplexity = float(math.exp(avg_loss)) if avg_loss < 50 else float("inf")
    iters_per_token = total_iters / max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "energy": avg_energy,
        "perplexity": perplexity,
        "iters_per_token": iters_per_token,
    }


def train_from_texts(texts: Iterable[str], config_name: str = "small", device: torch.device | None = None) -> UELM4:
    device = device or torch.device("cpu")
    cfg = load_config(config_name)
    model = UELM4(cfg).to(device)
    optimiser = AdamW(model.parameters(), lr=2e-4)
    tokenizer = SimpleTokenizer(vocab=set(" ".join(texts).split()))
    loader_cfg = LoaderConfig(max_length=cfg.solver.T_train * max(cfg.memory.shortlist_k, 4), batch_size=2)
    dataloader = build_dataloader(texts, tokenizer, loader_cfg)
    metrics = train_epoch(model, optimiser, dataloader, device)
    model.last_train_metrics = metrics
    return model


__all__ = ["train_epoch", "train_from_texts"]
