#!/usr/bin/env bash
# Modern-corpus capacity check (wider model + larger vocab) with energy reg (CPU)
set -euo pipefail

UNIEN_DIR="$HOME/Desktop/unien"
DATA_DIR="$UNIEN_DIR/data"
OUT_DIR="$HOME/Desktop/unien_out/modern_capacity_energy_$(date +%Y%m%d_%H%M%S)"

echo "[modern-capacity-energy] cd $UNIEN_DIR"
cd "$UNIEN_DIR"

if [ ! -d .venv ]; then
  echo "[modern-capacity-energy] creating .venv"
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

echo "[modern-capacity-energy] using corpus: $CORPUS"
# T=1 and T=2 with k=16, KL, beta_end=1.5; width=128; vocab=8192
python3 scripts/run_phase_b_grid.py \
  --output "$OUT_DIR" \
  --corpus "$CORPUS" \
  --tokens 2000000 \
  --seq-len 512 \
  --device cpu \
  --grid "T=1,2;k=16;wmf=0;be=1.5" \
  --width 128 \
  --vocab-size 8192 \
  --energy-reg 2e-4

echo "[modern-capacity-energy] opening report: $OUT_DIR/report.md"
open "$OUT_DIR/report.md" 2>/dev/null || true

