from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW

from ..config import load_config
from ..data.dataloaders import LoaderConfig, build_dataloader
from ..data.tokenization import SimpleTokenizer
from ..model.uelm4_model import UELM4
from ..train.losses import total_loss


def _forward_sequences(model: UELM4, batch: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.ndim == 1:
        logits = model(batch.to(device))
        return logits.unsqueeze(0), batch.unsqueeze(0).to(device)
    logits_list = []
    targets_list = []
    for seq in batch:
        seq = seq.to(device)
        logits_list.append(model(seq))
        targets_list.append(seq)
    return torch.stack(logits_list), torch.stack(targets_list)


def train_epoch(model: UELM4, optimiser: AdamW, dataloader: Iterable[torch.Tensor], device: torch.device) -> float:
    model.train()
    total = 0.0
    batches = 0
    for batch in dataloader:
        optimiser.zero_grad()
        logits, targets = _forward_sequences(model, batch, device)
        loss = total_loss(logits, targets)
        loss.backward()
        optimiser.step()
        total += float(loss.detach())
        batches += 1
    return total / max(batches, 1)


def train_from_texts(texts: Iterable[str], config_name: str = "small", device: torch.device | None = None) -> UELM4:
    device = device or torch.device("cpu")
    cfg = load_config(config_name)
    model = UELM4(cfg).to(device)
    optimiser = AdamW(model.parameters(), lr=2e-4)
    tokenizer = SimpleTokenizer(vocab=set(" ".join(texts).split()))
    loader_cfg = LoaderConfig(max_length=cfg.solver.T_train * max(cfg.memory.shortlist_k, 4), batch_size=2)
    dataloader = build_dataloader(texts, tokenizer, loader_cfg)
    train_epoch(model, optimiser, dataloader, device)
    return model


__all__ = ["train_epoch", "train_from_texts"]
