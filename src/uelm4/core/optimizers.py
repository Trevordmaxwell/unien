from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


Vector = torch.Tensor
MatVec = Callable[[Vector], Vector]


def conjugate_gradient(matvec: MatVec, b: Vector, x0: Vector | None = None, tol: float = 1e-6, max_iter: int = 32) -> Vector:
    """Solve ``Ax = b`` using the conjugate gradient method with implicit matvec."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r.flatten(), r.flatten())
    if rs_old <= tol:
        return x
    for _ in range(max_iter):
        Ap = matvec(p)
        alpha = rs_old / (torch.dot(p.flatten(), Ap.flatten()) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r.flatten(), r.flatten())
        if rs_new <= tol:
            break
        beta = rs_new / (rs_old + 1e-12)
        p = r + beta * p
        rs_old = rs_new
    return x


@dataclass
class AndersonState:
    history: list[Vector]
    weights: list[Vector]


def anderson_acceleration(new: Vector, state: AndersonState | None = None, m: int = 3, damping: float = 1.0) -> tuple[Vector, AndersonState]:
    """Apply a very small Anderson acceleration step to a fixed-point iterate."""
    if state is None:
        state = AndersonState(history=[], weights=[])
    history = (state.history + [new.detach()])[-m:]
    if len(history) < 2:
        state.history = history
        state.weights = []
        return new, state

    diffs = torch.stack([history[-1] - h for h in history[:-1]], dim=0)
    G = torch.matmul(diffs, diffs.transpose(0, 1)) + 1e-6 * torch.eye(len(diffs), device=new.device)
    rhs = torch.zeros(len(diffs), device=new.device)
    coeff = torch.linalg.solve(G, rhs)
    coeff = torch.cat([coeff, torch.tensor([1 - coeff.sum()], device=new.device)])
    accel = sum(w * h for w, h in zip(coeff, history))
    out = damping * new + (1 - damping) * accel
    state.history = history
    state.weights = [coeff]
    return out, state


__all__ = ["conjugate_gradient", "AndersonState", "anderson_acceleration"]
