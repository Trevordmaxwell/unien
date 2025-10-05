#!/usr/bin/env python
"""Stub ANN indexer for shortlist acceleration."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("landmarks", type=Path, help="Path to landmarks tensor")
    parser.add_argument("output", type=Path, help="Where to store the index")
    args = parser.parse_args()

    landmarks = torch.load(args.landmarks, map_location="cpu")
    norms = torch.norm(landmarks, dim=-1, keepdim=True)
    index = landmarks / norms.clamp_min(1e-9)
    torch.save(index, args.output)
    print(f"Saved ANN index with shape {index.shape} to {args.output}")


if __name__ == "__main__":
    main()
