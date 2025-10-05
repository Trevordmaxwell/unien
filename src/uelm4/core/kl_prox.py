import torch
import torch.nn.functional as F


def kl_masked_softmax(P: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    KL-prox on simplex face:
    p_new ∝ p_old * exp(η * scores). Here we set η=1 for Phase-A.
    Inputs:
      P:      (n,k) current probs (simplex per row)
      scores: (n,k) fit scores per shortlisted atom
      mask:   (n,k) boolean mask (optional), True for valid
    Returns:
      (n,k) new probs on simplex.
    """
    logits = torch.log(P.clamp_min(1e-9)) + scores
    if mask is not None:
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
    out = F.softmax(logits, dim=-1)
    return out
