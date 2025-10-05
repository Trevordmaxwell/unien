# UELM-4

Unified Equilibrium Language Model prototype covering Phase-A (table memory + KL prox) and scaffolding for Phase-B (CMM + WMF + controller).

## Features
- Dual-simplex state `P` with tied readout, causal shortlist, and cache-as-constraint support (identity/shift/EMA/conv advectors).
- Mirror-PDHG solver with optional Wasserstein-Mirror prox (iters/eps/cost scaling), meta-controller schedules, and ANN-accelerated shortlist search (FAISS fallback to cosine top-k).
- Causal symplectic/dissipative field implemented via reusable banded operators.
- Memory backends: table or Continuous Memory Measure (CMM) with landmark generator, metadata I/O, and RAG bridge hooks.
- Lightweight data, training, controller-distillation, and evaluation helpers plus profiling scripts.

## Layout
```
src/uelm4/
  config/        # YAML configs + loader
  core/          # solver, energy, cache, implicit autodiff, banded ops
  memory/        # shortlist, ANN index, table/CMM, readout, RAG bridge
  model/         # embeddings, scout, controller, decode
  data/          # tokenizer + dataloaders
  train/         # losses, schedules, controller distillation, training loops
scripts/          # build_landmarks.py, index_ann.py, profile_decode.py, bootstrap_phase_b.py
docs/             # design, API, experiment logs
```

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
```

Tiny CPU training loop:
```python
from uelm4.train import train_from_texts
model = train_from_texts(["hello world", "uelm4 prototype"], config_name="small")
```

Sample corpus training:
```bash
python scripts/train_sample.py --config small --epochs 2
```

Hyperparameter sweep:
```bash
python scripts/hparam_sweep.py --config small --epochs 1 --sweeps '[{"solver": {"T_train": 1}}, {"solver": {"T_train": 2}}]'
```

Controller distillation:
```python
from uelm4.config import load_config
from uelm4.data import SimpleTokenizer, build_dataloader, LoaderConfig
from uelm4.train import distill_controller
from uelm4.model import uelm4_model

cfg = load_config("small")
model = uelm4_model.UELM4(cfg)
tok = SimpleTokenizer(vocab={"hello", "world"})
loader = build_dataloader(["hello world"], tok, LoaderConfig(max_length=8, batch_size=1))
loss = distill_controller(model, loader, teacher_iters=3, student_iters=1)
```

Profile decode latency:
```bash
python scripts/profile_decode.py --config small --tokens 64 --new 16
```

Phase-B assets:
```bash
python scripts/bootstrap_phase_b.py path/to/table.pt artifacts/phase_b --num-landmarks 4096
python scripts/profile_decode.py --config small --landmarks artifacts/phase_b/landmarks.json --ann artifacts/phase_b/ann_index.json --use-wmf --tokens 128 --new 32
```

Documentation lives in `docs/`, configs in `src/uelm4/config/`, and reproducible scripts under `scripts/`.
