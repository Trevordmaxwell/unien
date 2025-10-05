import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d: int, max_len: int = 65536):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        nn.init.normal_(self.token.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.01)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (n,)
        n = tokens.shape[0]
        pos_ids = torch.arange(n, device=tokens.device)
        return self.token(tokens) + self.pos(pos_ids)
