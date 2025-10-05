import torch
import torch.nn as nn


class TiedReadout(nn.Module):
    """
    Tied readout: logits = Y @ M_lex^T + b
    M_lex is a (vocab_size, d) slice (or projection) of the memory table.
    """

    def __init__(self, M_lex: torch.Tensor, bias: bool = True, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.register_buffer("M_lex", M_lex)
        self.bias = nn.Parameter(torch.zeros(M_lex.shape[0])) if bias else None

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        logits = self.scale * (Y @ self.M_lex.t())
        if self.bias is not None:
            logits = logits + self.bias
        return logits
