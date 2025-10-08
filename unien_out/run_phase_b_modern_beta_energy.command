#!/usr/bin/env bash
# Modern-corpus KL beta scan with energy regularization (CPU)
set -euo pipefail

UNIEN_DIR="$HOME/Desktop/unien"
DATA_DIR="$UNIEN_DIR/data"
OUT_DIR="$HOME/Desktop/unien_out/modern_beta_energy_$(date +%Y%m%d_%H%M%S)"

echo "[modern-beta-energy] cd $UNIEN_DIR"
cd "$UNIEN_DIR"

if [ ! -d .venv ]; then
  echo "[modern-beta-energy] creating .venv"
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip >/dev/null || true
pip install -e . >/dev/null || true

mkdir -p "$DATA_DIR" "$OUT_DIR"
CORPUS="$DATA_DIR/corpus_wikitext103.txt"

if [ ! -s "$CORPUS" ]; then
  if [ -s "$DATA_DIR/corpus_wikitext2.txt" ]; then
    CORPUS="$DATA_DIR/corpus_wikitext2.txt"
  fi
fi

echo "[modern-beta-energy] using corpus: $CORPUS"
python3 scripts/run_phase_b_grid.py \
  --output "$OUT_DIR" \
  --corpus "$CORPUS" \
  --tokens 2000000 \
  --seq-len 512 \
  --device cpu \
  --grid "T=1,2,3;k=16,32;wmf=0;be=1.3,1.4,1.5,1.6" \
  --energy-reg 2e-4

echo "[modern-beta-energy] opening report: $OUT_DIR/report.md"
open "$OUT_DIR/report.md" 2>/dev/null || true

