#!/usr/bin/env python
"""Build an ANN index (FAISS if available, cosine fallback otherwise)."""

import argparse
import json
from pathlib import Path

import torch

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional
    faiss = None


def build_faiss_index(vectors: torch.Tensor, nlist: int, nprobe: int):
    if faiss is None:
        return None
    d = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    feats = vectors.detach().cpu().numpy().astype("float32")
    index.train(feats)
    index.add(feats)
    index.nprobe = nprobe
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("landmarks", type=Path, help="Path to landmark tensor (.pt)")
    parser.add_argument("output", type=Path, help="Base path for index metadata (e.g. index.json)")
    parser.add_argument("--nlist", type=int, default=128, help="Number of IVF lists (FAISS)")
    parser.add_argument("--nprobe", type=int, default=8, help="Number of probes per query")
    args = parser.parse_args()

    landmarks = torch.load(args.landmarks, map_location="cpu")
    if landmarks.ndim != 2:
        raise ValueError("Landmarks tensor must be of shape (K, d)")

    meta = {"vectors": str(args.landmarks)}
    if faiss is not None:
        faiss_index = build_faiss_index(landmarks, args.nlist, args.nprobe)
        faiss_path = args.output.with_suffix(".faiss")
        faiss.write_index(faiss_index, str(faiss_path))
        meta.update({
            "type": "faiss",
            "index_path": faiss_path.name,
            "nlist": args.nlist,
            "nprobe": args.nprobe,
        })
    else:
        meta.update({"type": "cosine", "normalize": True})
    meta_path = args.output if args.output.suffix else args.output.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved ANN metadata to {meta_path}")


if __name__ == "__main__":
    main()
