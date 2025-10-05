from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch


@dataclass
class TokenizerConfig:
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"


class SimpleTokenizer:
    """Very small whitespace tokenizer used for synthetic tests."""

    def __init__(self, vocab: Iterable[str] | None = None, cfg: TokenizerConfig | None = None):
        cfg = cfg or TokenizerConfig()
        special = [cfg.unk_token, cfg.pad_token, cfg.bos_token, cfg.eos_token]
        vocab = list(vocab) if vocab is not None else []
        vocab = special + [tok for tok in vocab if tok not in special]
        self.token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.cfg = cfg

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        pieces = text.strip().split()
        ids = [self.token_to_id.get(piece, self.token_to_id[self.cfg.unk_token]) for piece in pieces]
        if add_special:
            ids = [self.token_to_id[self.cfg.bos_token]] + ids + [self.token_to_id[self.cfg.eos_token]]
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        tokens: List[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.cfg.unk_token)
            if skip_special and token in {self.cfg.bos_token, self.cfg.eos_token, self.cfg.pad_token}:
                continue
            tokens.append(token)
        return " ".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def to_tensor(self, text: str) -> torch.Tensor:
        ids = self.encode(text)
        return torch.tensor(ids, dtype=torch.long)


__all__ = ["SimpleTokenizer", "TokenizerConfig"]
