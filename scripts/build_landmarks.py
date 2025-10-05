#!/usr/bin/env python
"""Utility to build Nyström landmarks from a memory table."""

import argparse
from pathlib import Path

import torch

from uelm4.memory.landmark_io import save_landmarks
from uelm4.memory.landmarks import select_landmarks

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("memory", type=Path, help="Path to .pt tensor containing the memory table")
    parser.add_argument("output", type=Path, help="Destination base path for landmarks (without extension)")
    parser.add_argument("--num", type=int, default=1024, help="Number of landmarks to select")
    args = parser.parse_args()

    table = torch.load(args.memory, map_location="cpu")
    _, landmarks = select_landmarks(table, args.num)
    tensor_path, meta_path = save_landmarks(landmarks, args.output)
    print(f"Saved landmarks to {tensor_path} with metadata {meta_path}")


if __name__ == "__main__":
    main()
