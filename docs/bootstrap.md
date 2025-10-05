# Phase-B Bootstrap Guide

Use `scripts/bootstrap_phase_b.py` to materialize Phase-B assets (landmarks + ANN index) from a table memory tensor:

```bash
python scripts/bootstrap_phase_b.py path/to/table.pt artifacts/phase_b --num-landmarks 4096 --ann-nlist 256 --ann-nprobe 16
```

This produces:
- `artifacts/phase_b/landmarks.pt` and `landmarks.json`
- `artifacts/phase_b/ann_index.json` (+ optional `ann_index.faiss` when FAISS is available)

Point the config to:

```
memory:
  type: "cmm"
  meta_path: artifacts/phase_b/landmarks.json
```

and set `ann_index` when calling the model (path to `ann_index.json`).

Quick profile using the saved assets:

```bash
python scripts/profile_decode.py --config small --landmarks artifacts/phase_b/landmarks.json --ann artifacts/phase_b/ann_index.json --use-wmf --tokens 128 --new 32
```
