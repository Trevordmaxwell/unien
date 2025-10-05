# UELM-4 Design Notes

This document sketches the Phase-A/B architecture:

- **State**: simplex `P`, representation `Y`, dual `Λ`.
- **Solver**: Mirror-PDHG with optional Wasserstein prox and controller schedules.
- **Field**: causal symplectic/dissipative split implemented via banded convolutions.
- **Memory**: table memory for Phase-A, continuous memory measure (CMM) for Phase-B.
- **Caches**: shortlist, cache-as-constraint, tied readout to lexical slice.

Future work includes Cache-as-Constraint tuning, WMF stabilisation, and controller distillation.
