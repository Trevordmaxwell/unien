import torch


class IdentityPrecond:
    def apply(self, grad: torch.Tensor) -> torch.Tensor:
        return grad
