# UELM-4 (Phase-A MVP)

Minimal, testable implementation of the Unified Equilibrium Language Model:
- Dual-Simplex state `P` (top-k per token), representation `Y = M^T P`
- Causal banded field (symplectic-dissipative split, simplified)
- Table memory with per-token shortlist
- KL-prox update on the simplex (masked softmax)
- Two-block solver step (Y-step + P-step + Dual)
- Tied readout from `Y` to logits

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest
```

See `configs/base16k.yaml` for default settings.
