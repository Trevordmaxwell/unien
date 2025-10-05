# Public API Overview

```python
from uelm4.config import load_config
from uelm4.model.uelm4_model import UELM4
from uelm4.model.decode import greedy_decode
from uelm4.memory.ann import build_ann_index
from uelm4.train import distill_controller, train_from_texts

cfg = load_config("small")
model = UELM4(cfg)
logits = model(tokens)
ids = greedy_decode(model, prompt_ids)
index = build_ann_index(model.memory.detach() if hasattr(model.memory, 'detach') else model.memory)
loss = distill_controller(model, [tokens], teacher_iters=3, student_iters=1)
model = train_from_texts(["hello world"], config_name="small")
print(model.last_train_metrics)
```

Helper modules:

- `uelm4.data` – lightweight tokenizer and dataloader helpers.
- `uelm4.train` – loss functions, schedules, controller distillation, and training utilities.
- `uelm4.core` – solver, energy, cache, implicit differentiation, banded ops, ANN support.
