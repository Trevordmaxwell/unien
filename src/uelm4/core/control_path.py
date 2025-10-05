from dataclasses import dataclass
import torch


@dataclass
class Path:
    times: torch.Tensor   # (n,)
    knots: torch.Tensor   # (n, dx)


def make_control_path(E: torch.Tensor) -> Path:
    """
    Build a simple piecewise-linear control path from embeddings.
    For Phase-A: dx == d and knots := E.
    """
    n, _ = E.shape
    times = torch.linspace(0.0, 1.0, n, device=E.device, dtype=E.dtype)
    return Path(times=times, knots=E)
