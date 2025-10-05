# UELM-4

Unified Equilibrium Language Model prototype covering Phase-A (table memory + KL prox) and scaffolding for Phase-B (CMM + WMF + controller).

## Features
- Dual-simplex state `P` with tied readout and causal shortlist.
- Mirror-PDHG solver with optional Wasserstein-Mirror prox and meta-controller schedules.
- Causal symplectic/dissipative field implemented via reusable banded operators.
- Memory backends: table or Continuous Memory Measure (CMM) with landmark generator.
- Lightweight data, training, and evaluation helpers plus profiling scripts.

## Layout
```
src/uelm4/
  config/        # YAML configs + loader
  core/          # solver, energy, implicit autodiff, banded ops
  memory/        # shortlist, table/CMM, readout, RAG bridge
  model/         # embeddings, scout, controller, decode
  data/          # tokenizer + dataloaders
  train/         # losses, schedules, training loops
scripts/          # build_landmarks.py, index_ann.py, profile_decode.py
docs/             # design, API, experiment logs
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
```

To run a tiny training loop on CPU:
```python
from uelm4.train import train_from_texts
model = train_from_texts(["hello world", "uelm4 prototype"], config_name="small")
```

Profile decode latency:
```bash
python scripts/profile_decode.py --config small --tokens 64 --new 16
```

Documentation lives in `docs/`, configs in `src/uelm4/config/`, and reproducible scripts under `scripts/`.
