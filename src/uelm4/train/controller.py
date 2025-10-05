from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch.optim import Adam

from ..model.uelm4_model import UELM4


def _iterate_sequences(batch: torch.Tensor) -> list[torch.Tensor]:
    if batch.ndim == 1:
        return [batch]
    return [row for row in batch]


def distill_controller(
    model: UELM4,
    batches: Iterable[torch.Tensor],
    teacher_iters: int = 3,
    student_iters: int = 1,
    lr: float = 1e-3,
    device: torch.device | None = None,
    force_full_iters: bool = True,
) -> float:
    """One epoch of controller distillation using a teacher solve."""
    if not hasattr(model.solver, "controller") or model.solver.controller is None:
        raise ValueError("Model does not have an attached controller to distill")

    controller = model.solver.controller
    device = device or next(model.parameters()).device
    optimiser = Adam(controller.parameters(), lr=lr)
    model.train()
    total_loss = 0.0
    steps = 0
    original_tol = model.cfg.solver.early_exit_tol
    if force_full_iters:
        model.cfg.solver.early_exit_tol = -float("inf")
    try:
        for batch in batches:
            seqs = _iterate_sequences(batch)
            for seq in seqs:
                optimiser.zero_grad()
                tokens = seq.to(device)
                with torch.no_grad():
                    teacher_logits = model(tokens, T=teacher_iters).detach()
                student_logits = model(tokens, T=student_iters)
                loss = F.mse_loss(student_logits, teacher_logits)
                loss.backward()
                optimiser.step()
                total_loss += float(loss.detach())
                steps += 1
    finally:
        model.cfg.solver.early_exit_tol = original_tol
    return total_loss / max(steps, 1)


__all__ = ["distill_controller"]
