from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenization import SimpleTokenizer


@dataclass
class LoaderConfig:
    max_length: int = 512
    batch_size: int = 4
    shuffle: bool = True


class PackedDataset(Dataset):
    """Dataset that packs tokenized sequences to a fixed length."""

    def __init__(self, texts: Iterable[str], tokenizer: SimpleTokenizer, max_length: int):
        self.texts: List[str] = list(texts)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        tokens = self.tok.to_tensor(self.texts[idx])
        if tokens.numel() >= self.max_length:
            return tokens[: self.max_length]
        pad_len = self.max_length - tokens.numel()
        pad_id = self.tok.token_to_id[self.tok.cfg.pad_token]
        padding = torch.full((pad_len,), pad_id, dtype=torch.long)
        return torch.cat([tokens, padding], dim=0)


def build_dataloader(texts: Iterable[str], tokenizer: SimpleTokenizer, cfg: LoaderConfig) -> DataLoader:
    dataset = PackedDataset(texts, tokenizer, cfg.max_length)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle)


__all__ = ["LoaderConfig", "PackedDataset", "build_dataloader"]
