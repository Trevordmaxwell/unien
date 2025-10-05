from __future__ import annotations

from typing import Iterable

import torch
from torch.optim import AdamW

from ..config import load_config
from ..data.dataloaders import LoaderConfig, build_dataloader
from ..data.tokenization import SimpleTokenizer
from ..model.uelm4_model import UELM4
from ..train.losses import total_loss


def train_epoch(model: UELM4, optimiser: AdamW, dataloader: Iterable[torch.Tensor], device: torch.device) -> float:
    model.train()
    total = 0.0
    for batch in dataloader:
        optimiser.zero_grad()
        batch = batch.to(device)
        logits = model(batch)
        loss = total_loss(logits, batch)
        loss.backward()
        optimiser.step()
        total += float(loss.detach())
    return total / max(len(dataloader), 1)


def train_from_texts(texts: Iterable[str], config_name: str = "small", device: torch.device | None = None) -> UELM4:
    device = device or torch.device("cpu")
    cfg = load_config(config_name)
    model = UELM4(cfg).to(device)
    optimiser = AdamW(model.parameters(), lr=2e-4)
    tokenizer = SimpleTokenizer(vocab=set(" ".join(texts).split()))
    loader_cfg = LoaderConfig(max_length=cfg.solver.T_train * cfg.memory.shortlist_k, batch_size=2)
    dataloader = build_dataloader(texts, tokenizer, loader_cfg)
    train_epoch(model, optimiser, dataloader, device)
    return model


__all__ = ["train_epoch", "train_from_texts"]
