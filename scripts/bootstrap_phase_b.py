#!/usr/bin/env python
"""Bootstrap script for Phase-B assets (landmarks + ANN index)."""

import argparse
from pathlib import Path

import torch

from uelm4.memory.bootstrap import build_phase_b_assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("memory", type=Path, help="Path to .pt tensor containing the memory table")
    parser.add_argument("out", type=Path, help="Output directory for assets")
    parser.add_argument("--num-landmarks", type=int, default=2048)
    parser.add_argument("--ann-nlist", type=int, default=128)
    parser.add_argument("--ann-nprobe", type=int, default=8)
    args = parser.parse_args()

    table = torch.load(args.memory, map_location="cpu")
    lm_meta, ann_meta = build_phase_b_assets(
        table,
        args.out,
        num_landmarks=args.num_landmarks,
        ann_nlist=args.ann_nlist,
        ann_nprobe=args.ann_nprobe,
    )
    print(f"Landmark metadata -> {lm_meta}")
    print(f"ANN metadata -> {ann_meta}")


if __name__ == "__main__":
    main()
