import torch

from .uelm4_model import UELM4


@torch.no_grad()
def greedy_decode(model: UELM4, prompt_ids: torch.Tensor, max_new_tokens: int = 32):
    ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(ids)              # (n, vocab)
        next_id = logits[-1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=0)
    return ids
