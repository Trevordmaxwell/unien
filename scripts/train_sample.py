#!/usr/bin/env python
"""Train the UELM-4 model on a tiny Shakespeare excerpt."""

import argparse
from pathlib import Path

import torch

from uelm4.config import load_config
from uelm4.data import LoaderConfig, SimpleTokenizer, build_dataloader
from uelm4.train import train_epoch
from uelm4.model.uelm4_model import UELM4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="small", help="Config to load")
    parser.add_argument("--corpus", type=Path, default=Path("data/tiny_shakespeare_excerpt.txt"))
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    text = args.corpus.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    model = UELM4(cfg).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4)

    vocab = set(" ".join(lines).split())
    tokenizer = SimpleTokenizer(vocab=vocab)
    loader_cfg = LoaderConfig(max_length=cfg.solver.T_train * max(cfg.memory.shortlist_k, 4), batch_size=2, shuffle=True)
    dataloader = build_dataloader(lines, tokenizer, loader_cfg)

    for epoch in range(args.epochs):
        metrics = train_epoch(model, optimiser, dataloader, device)
        print(
            "epoch={epoch} loss={loss:.4f} ppl={ppl:.2f} energy={energy:.4f} iters/token={iters:.2f}"
            .format(
                epoch=epoch + 1,
                loss=metrics["loss"],
                ppl=metrics["perplexity"],
                energy=metrics["energy"],
                iters=metrics["iters_per_token"],
            )
        )

    lm_head = model.readout if hasattr(model, "readout") else None
    if lm_head is not None:
        out_dir = Path("artifacts")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "uelm4_shakespeare.pt")


if __name__ == "__main__":
    main()
