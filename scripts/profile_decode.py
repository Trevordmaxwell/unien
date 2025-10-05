#!/usr/bin/env python
"""Profile the autoregressive decode loop."""

import argparse
import time

import torch

from uelm4.config import load_config
from uelm4.model.decode import greedy_decode
from uelm4.model.uelm4_model import UELM4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="small", help="Config name to load")
    parser.add_argument("--tokens", type=int, default=64, help="Prompt length")
    parser.add_argument("--new", type=int, default=16, help="Tokens to generate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    model = UELM4(cfg).to(device).eval()
    prompt = torch.randint(0, cfg.model.vocab_size, (args.tokens,), device=device)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start = time.perf_counter()
    greedy_decode(model, prompt, max_new_tokens=args.new)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"Generated {args.new} tokens in {elapsed*1000:.2f} ms")


if __name__ == "__main__":
    main()
