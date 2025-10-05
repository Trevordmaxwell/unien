# Public API Overview

```python
from uelm4.config import load_config
from uelm4.model.uelm4_model import UELM4
from uelm4.model.decode import greedy_decode

cfg = load_config("small")
model = UELM4(cfg)
logits = model(tokens)
ids = greedy_decode(model, prompt_ids)
```

Helper modules:

- `uelm4.data` – lightweight tokenizer and dataloader helpers.
- `uelm4.train` – loss functions, metrics, and utility schedules.
- `uelm4.core` – solver, energy, banded ops, implicit differentiation.
