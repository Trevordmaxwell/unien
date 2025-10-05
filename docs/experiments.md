# Experiment Tracking

| Experiment | Config | Notes |
|------------|--------|-------|
| phase-a-mvp | `small` | KL prox, table memory |
| phase-b-proto | `base16k` (WMF) | Enable `solver.use_wmf` and switch to CMM |

Record `pytest -q` output and profiler traces in this file alongside seed hashes.
