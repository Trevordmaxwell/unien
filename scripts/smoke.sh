#!/usr/bin/env bash
set -euo pipefail

python3 scripts/run_phase_b_grid.py --output tmp_grid --tokens 200000 --seq-len 128 --device cpu

python3 - <<'PY'
from pathlib import Path
import json, sys
req = {"label","config","final_loss","perplexity","final_energy","iters_per_token","decode_ms_per_token","forward_ms_per_token","total_seconds","tokens_seen","epochs","hardware"}
p = Path('tmp_grid')/'grid_results.json'
if not p.exists():
    sys.stderr.write('grid_results.json not found\n')
    sys.exit(1)
runs = json.loads(p.read_text())
if not isinstance(runs, list) or not runs:
    sys.stderr.write('grid_results.json must be a non-empty list\n')
    sys.exit(1)
bad = []
for i, r in enumerate(runs):
    missing = req - set(r.keys())
    if missing:
        bad.append((i, sorted(missing)))
if bad:
    for i, keys in bad:
        sys.stderr.write(f'Run {i} missing keys: {keys}\n')
    sys.exit(1)
print('Smoke OK: grid_results contains required keys')
PY
