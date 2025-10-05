import math
import torch

from uelm4.config import load_config
from uelm4.model.uelm4_model import UELM4
from uelm4.train.controller import distill_controller


def test_controller_distillation_runs():
    cfg = load_config("small")
    model = UELM4(cfg)
    batches = [torch.randint(0, cfg.model.vocab_size, (16,)) for _ in range(3)]
    loss = distill_controller(model, batches, teacher_iters=3, student_iters=1, lr=1e-2)
    assert math.isfinite(loss)
    assert loss >= 0.0
