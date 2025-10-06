#!/usr/bin/env python3
"""Prepare a simple line-based corpus from text or JSONL sources.

Usage examples:
  python scripts/prepare_corpus.py --input /path/to/folder --output data/corpus.txt
  python scripts/prepare_corpus.py --input big.jsonl --jsonl-key text --output data/corpus.txt

Notes:
  - Produces one line per example (trimmed, empty lines removed).
  - Keeps only basic ASCII whitespace normalization; no aggressive filtering.
  - Designed to work with the SimpleTokenizer used in the grid script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_text_files(root: Path) -> Iterable[str]:
    # Include common plain-text extensions for public corpora
    exts = {".txt", ".md", ".log", ".raw", ".tokens"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            try:
                yield p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue


def iter_jsonl(path: Path, key: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            txt = obj.get(key) or obj.get("content") or obj.get("text")
            if isinstance(txt, str):
                yield txt


def to_lines(texts: Iterable[str]) -> list[str]:
    out: list[str] = []
    for t in texts:
        for line in t.splitlines():
            s = " ".join(line.strip().split())
            if s:
                out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Input file/folder (txt/md/log or JSONL)")
    ap.add_argument("--output", type=Path, required=True, help="Output corpus file (one example per line)")
    ap.add_argument("--jsonl-key", type=str, default="text", help="Key to read from JSONL objects (default: text)")
    args = ap.parse_args()

    src = args.input
    if src.is_dir():
        texts = iter_text_files(src)
    elif src.suffix.lower() in {".jsonl", ".jl"}:
        texts = iter_jsonl(src, args.jsonl_key)
    else:
        texts = [src.read_text(encoding="utf-8", errors="ignore")]

    lines = to_lines(texts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} lines -> {args.output}")


if __name__ == "__main__":
    main()
