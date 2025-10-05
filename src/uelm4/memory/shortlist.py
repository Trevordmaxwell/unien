import torch


def shortlist(E: torch.Tensor, M: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simple cosine-sim shortlist per token (Phase-A, CPU/GPU friendly).
    E: (n,d) token embeddings
    M: (K,d) memory table
    Returns:
      Kset: (n,k) long indices
      mask: (n,k) bool (all True here; placeholder for later constraints)
    """
    n, _ = E.shape
    K = M.shape[0]
    # Normalize for cosine
    En = E / (E.norm(dim=-1, keepdim=True) + 1e-9)
    Mn = M / (M.norm(dim=-1, keepdim=True) + 1e-9)
    scores = En @ Mn.t()                       # (n,K)
    topk = torch.topk(scores, k=min(k, K), dim=-1)
    Kset = topk.indices.long()
    mask = torch.ones_like(Kset, dtype=torch.bool, device=Kset.device)
    return Kset, mask
