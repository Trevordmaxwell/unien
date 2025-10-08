#!/usr/bin/env bash
# Long CPU run (modern corpus): capacity + longer budget
# WARNING: This can take many hours on CPU.
set -euo pipefail

UNIEN_DIR="$HOME/Desktop/unien"
DATA_DIR="$UNIEN_DIR/data"
OUT_DIR="$HOME/Desktop/unien_out/modern_long_cpu_$(date +%Y%m%d_%H%M%S)"

echo "[modern-long-cpu] cd $UNIEN_DIR"
cd "$UNIEN_DIR"

if [ ! -d .venv ]; then
  echo "[modern-long-cpu] creating .venv"
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip >/dev/null || true
pip install -e . >/dev/null || true

mkdir -p "$DATA_DIR" "$OUT_DIR"
CORPUS="$DATA_DIR/corpus_wikitext103.txt"
if [ ! -s "$CORPUS" ] && [ -s "$DATA_DIR/corpus_wikitext2.txt" ]; then
  CORPUS="$DATA_DIR/corpus_wikitext2.txt"
fi

echo "[modern-long-cpu] using corpus: $CORPUS"
# Fast capacity preset with larger budget
python3 scripts/run_phase_b_grid.py \
  --output "$OUT_DIR" \
  --corpus "$CORPUS" \
  --tokens 20000000 \
  --seq-len 512 \
  --device cpu \
  --grid "T=1;k=32;wmf=0;be=1.5" \
  --width 128 \
  --vocab-size 8192 \
  --energy-reg 2e-4

echo "[modern-long-cpu] opening report: $OUT_DIR/report.md"
open "$OUT_DIR/report.md" 2>/dev/null || true

