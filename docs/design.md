# UELM-4 Design Notes

This document sketches the Phase-A/B architecture:

- **State**: simplex `P`, representation `Y`, dual `Λ`, and cache state `C` used by Cache-as-Constraint (CaC). CaC now supports identity, shift, EMA, and causal convolution advectors to pin history, delay, exponentially smooth, or filter trajectories.
- **Solver**: Mirror-PDHG with optional Wasserstein prox (tunable iterations/eps/cost scaling), controller-driven schedules, and CaC penalty injection with energy accounting.
- **Field**: causal symplectic/dissipative split implemented via banded convolutions.
- **Memory**: table memory for Phase-A, continuous memory measure (CMM) for Phase-B, plus ANN shortlist acceleration.
- **Caches**: shortlist, CaC, tied readout to lexical slice; ANN index may be pre-built with FAISS.

Future work includes WMF stabilisation, richer advectors beyond EMA/shift, and controller distillation on large corpora.
