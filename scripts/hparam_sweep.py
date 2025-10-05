#!/usr/bin/env python
"""Run a lightweight hyperparameter sweep on a small corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uelm4.train.hparam_sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="small", help="Base config name")
    parser.add_argument("--corpus", type=Path, default=Path("data/tiny_shakespeare_excerpt.txt"))
    parser.add_argument(
        "--sweeps",
        required=True,
        help=(
            "JSON list of override dicts, e.g. "
            "[{\"solver\": {\"T_train\": 1}}, {\"solver\": {\"T_train\": 2}}]"
        ),
    )
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    sweeps = json.loads(args.sweeps)
    results = run_sweep(args.config, args.corpus, sweeps, epochs=args.epochs)
    for res in results:
        override_str = json.dumps(res.config_overrides)
        print(f"overrides={override_str} => loss={res.metrics['loss']:.4f} energy={res.metrics['energy']:.4f}")


if __name__ == "__main__":
    main()
