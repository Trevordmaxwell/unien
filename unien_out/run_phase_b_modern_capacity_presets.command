#!/usr/bin/env bash
# Capacity-boosted modern presets (Fast + Balanced) with energy regularization (CPU)
set -euo pipefail

UNIEN_DIR="$HOME/Desktop/unien"
DATA_DIR="$UNIEN_DIR/data"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_FAST="$HOME/Desktop/unien_out/modern_capacity_presets_${STAMP}_fast"
OUT_BAL="$HOME/Desktop/unien_out/modern_capacity_presets_${STAMP}_balanced"

echo "[modern-capacity-presets] cd $UNIEN_DIR"
cd "$UNIEN_DIR"

if [ ! -d .venv ]; then
  echo "[modern-capacity-presets] creating .venv"
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip >/dev/null || true
pip install -e . >/dev/null || true

mkdir -p "$DATA_DIR" "$OUT_FAST" "$OUT_BAL"
CORPUS="$DATA_DIR/corpus_wikitext103.txt"

if [ ! -s "$CORPUS" ]; then
  # Fallback to WikiText-2 if present
  if [ -s "$DATA_DIR/corpus_wikitext2.txt" ]; then
    CORPUS="$DATA_DIR/corpus_wikitext2.txt"
  fi
fi

echo "[modern-capacity-presets] using corpus: $CORPUS"

# Fast (capacity-boosted): T=1, k=32, KL, be=1.5, d=128, vocab=8192
python3 scripts/run_phase_b_grid.py \
  --output "$OUT_FAST" \
  --corpus "$CORPUS" \
  --tokens 2000000 \
  --seq-len 512 \
  --device cpu \
  --grid "T=1;k=32;wmf=0;be=1.5" \
  --width 128 \
  --vocab-size 8192 \
  --energy-reg 2e-4

# Balanced (capacity-boosted): T=2, k=16, KL, be=1.4, d=128, vocab=8192
python3 scripts/run_phase_b_grid.py \
  --output "$OUT_BAL" \
  --corpus "$CORPUS" \
  --tokens 2000000 \
  --seq-len 512 \
  --device cpu \
  --grid "T=2;k=16;wmf=0;be=1.4" \
  --width 128 \
  --vocab-size 8192 \
  --energy-reg 2e-4

echo "[modern-capacity-presets] opening reports"
open "$OUT_FAST/report.md" 2>/dev/null || true
open "$OUT_BAL/report.md" 2>/dev/null || true

