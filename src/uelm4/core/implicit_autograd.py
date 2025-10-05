from __future__ import annotations

from typing import Callable

import torch

from .optimizers import conjugate_gradient


FixedPointFn = Callable[[torch.Tensor], torch.Tensor]


def fixed_point_solve(fn: FixedPointFn, x0: torch.Tensor, max_iter: int = 16, tol: float = 1e-6) -> torch.Tensor:
    """Simple fixed-point iteration with stopping tolerance."""
    x = x0
    for _ in range(max_iter):
        x_next = fn(x)
        if torch.norm(x_next - x) <= tol:
            x = x_next
            break
        x = x_next
    return x.detach()


def implicit_grad(fn: FixedPointFn, x_star: torch.Tensor, grad_output: torch.Tensor, tol: float = 1e-6, max_iter: int = 16) -> torch.Tensor:
    """Jacobian-free backward pass for implicit function solves."""
    x_star = x_star.detach().requires_grad_(True)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        (jvp,) = torch.autograd.functional.jvp(fn, (x_star,), (v,), create_graph=True)
        return v - jvp

    solution = conjugate_gradient(matvec, grad_output, tol=tol, max_iter=max_iter)
    return solution.detach()


__all__ = ["fixed_point_solve", "implicit_grad"]
