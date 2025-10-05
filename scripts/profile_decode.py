#!/usr/bin/env python
"""Profile the autoregressive decode loop."""

import argparse
import time
from pathlib import Path

import torch

from uelm4.config import load_config
from uelm4.model.decode import greedy_decode
from uelm4.model.uelm4_model import UELM4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="small", help="Config name to load")
    parser.add_argument("--tokens", type=int, default=64, help="Prompt length")
    parser.add_argument("--new", type=int, default=16, help="Tokens to generate")
    parser.add_argument(
        "--landmarks",
        type=Path,
        help="Optional path to landmark metadata (landmarks.json) for Phase-B runs",
    )
    parser.add_argument(
        "--ann",
        type=Path,
        help="Optional path to ANN metadata (ann_index.json) produced by bootstrap",
    )
    parser.add_argument(
        "--use-wmf",
        action="store_true",
        help="Enable WMF solver settings (requires landmarks).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = {}
    if args.landmarks is not None:
        overrides.setdefault("memory", {})["type"] = "cmm"
        overrides["memory"]["meta_path"] = str(args.landmarks)
    if args.use_wmf:
        overrides.setdefault("solver", {})["use_wmf"] = True
    cfg = load_config(args.config, overrides)
    model = UELM4(cfg).to(device).eval()
    prompt = torch.randint(0, cfg.model.vocab_size, (args.tokens,), device=device)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start = time.perf_counter()
    ann_arg = str(args.ann) if args.ann is not None else None
    greedy_decode(model, prompt, max_new_tokens=args.new, ann_index=ann_arg)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"Generated {args.new} tokens in {elapsed*1000:.2f} ms")


if __name__ == "__main__":
    main()
