from __future__ import annotations


def linear_schedule(step: int, total: int, start: float, end: float) -> float:
    if total <= 1:
        return end
    alpha = min(max(step / (total - 1), 0.0), 1.0)
    return start + alpha * (end - start)


def cosine_schedule(step: int, total: int, start: float, end: float) -> float:
    import math

    if total <= 1:
        return end
    alpha = min(max(step / (total - 1), 0.0), 1.0)
    cos_val = (1 + math.cos(math.pi * (1 - alpha))) / 2
    return end + (start - end) * cos_val


__all__ = ["linear_schedule", "cosine_schedule"]
