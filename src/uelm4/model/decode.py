import torch

from ..memory.ann import ANNIndex
from .uelm4_model import UELM4


@torch.no_grad()
def greedy_decode(
    model: UELM4,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 32,
    ann_index: ANNIndex | str | None = None,
):
    ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(ids, ann_index=ann_index)
        next_id = logits[-1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=0)
    return ids
