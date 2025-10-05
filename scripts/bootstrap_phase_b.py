#!/usr/bin/env python
"""Bootstrap script for Phase-B assets: landmarks + ANN index."""

import argparse
from pathlib import Path

import torch

from uelm4.memory.landmark_io import save_landmarks
from uelm4.memory.landmarks import select_landmarks
from uelm4.memory.ann import build_ann_index


def build_assets(memory_path: Path, out_dir: Path, num_landmarks: int, ann_nlist: int, ann_nprobe: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = torch.load(memory_path, map_location="cpu")
    idx, landmarks = select_landmarks(table, num_landmarks)
    lm_tensor_path, lm_meta_path = save_landmarks(landmarks, out_dir / "landmarks")
    ann_index = build_ann_index(landmarks, nlist=ann_nlist, nprobe=ann_nprobe)
    ann_meta = {
        "type": "faiss" if ann_index.index is not None else "cosine",
        "normalize": ann_index.normalize,
        "nlist": ann_nlist,
        "nprobe": ann_nprobe,
        "path": "ann_index.faiss" if ann_index.index is not None else "",
    }
    ann_meta_path = out_dir / "ann_index.json"
    with ann_meta_path.open("w", encoding="utf-8") as fh:
        import json

        json.dump(ann_meta, fh, indent=2)
    if ann_index.index is not None:
        from faiss import write_index  # type: ignore

        write_index(ann_index.index, str(out_dir / ann_meta["path"]))
    print(f"Landmarks -> {lm_tensor_path}, metadata -> {lm_meta_path}")
    print(f"ANN index metadata -> {ann_meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("memory", type=Path, help="Path to memory table tensor (.pt)")
    parser.add_argument("out", type=Path, help="Output directory for assets")
    parser.add_argument("--num-landmarks", type=int, default=2048)
    parser.add_argument("--ann-nlist", type=int, default=128)
    parser.add_argument("--ann-nprobe", type=int, default=8)
    args = parser.parse_args()
    build_assets(args.memory, args.out, args.num_landmarks, args.ann_nlist, args.ann_nprobe)


if __name__ == "__main__":
    main()
