#!/usr/bin/env python
"""Run an extended Phase-B sweep and log detailed metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from itertools import product
from dataclasses import asdict
import platform
from pathlib import Path
from typing import Iterable

import torch

from uelm4.config import load_config
from uelm4.data import LoaderConfig, SimpleTokenizer, build_dataloader
from uelm4.model.decode import greedy_decode
from uelm4.core.energy import compute_energy_terms, total_energy
from uelm4.model.uelm4_model import UELM4
from uelm4.train.train import train_epoch


def ensure_corpus(path: Path) -> list[str]:
    if not path.exists():
        DEFAULT_LINES = (
            "To be, or not to be: that is the question.\n",
            "Whether 'tis nobler in the mind to suffer\n",
            "The slings and arrows of outrageous fortune,\n",
            "Or to take arms against a sea of troubles,\n",
            "And by opposing end them.\n",
        )
        text = "".join(DEFAULT_LINES)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        # Fallback to a small default corpus if the provided file is empty
        fallback = (
            "hello world from uelm\n",
            "this is a fallback modern corpus line\n",
            "language modeling with equilibrium dynamics\n",
            "shortlist and kl proximal updates are efficient\n",
        )
        return [s.strip() for s in fallback]
    return lines


def _top_k_vocab_from_lines(lines: Iterable[str], limit: int) -> list[str]:
    from collections import Counter
    tokens = []
    for line in lines:
        tokens.extend(line.strip().split())
    if limit <= 0:
        return []
    counts = Counter(tokens)
    return [tok for tok, _ in counts.most_common(limit)]


def build_dataloader_from_lines(lines: Iterable[str], cfg, seq_len: int, batch_size: int = 2) -> torch.utils.data.DataLoader:
    # Cap tokenizer vocab to model.vocab_size (reserve 4 for special tokens)
    max_vocab = max(int(cfg.model.vocab_size) - 4, 0)
    vocab_list = _top_k_vocab_from_lines(lines, max_vocab)
    tokenizer = SimpleTokenizer(vocab=vocab_list)
    loader_cfg = LoaderConfig(max_length=seq_len, batch_size=batch_size, shuffle=True)
    return build_dataloader(list(lines), tokenizer, loader_cfg)


def profile_forward(model: UELM4, prompt: torch.Tensor) -> tuple[float, float]:
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        _, state, _ = model(prompt, return_state=True)
    elapsed = time.perf_counter() - start
    iters = getattr(state, "iters", 0.0)
    iters_per_token = float(iters) / max(int(prompt.numel()), 1)
    return elapsed * 1000, iters_per_token


def profile_decode(model: UELM4, prompt: torch.Tensor, new_tokens: int = 32) -> float:
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        greedy_decode(model, prompt.clone(), max_new_tokens=new_tokens)
    elapsed = time.perf_counter() - start
    return (elapsed * 1000) / new_tokens


def forward_ms_per_token(model: UELM4, vocab_size: int, seq_len: int = 256, reps: int = 3, device: torch.device | None = None) -> tuple[float, float]:
    device = device or next(model.parameters()).device
    total_ms = 0.0
    total_iters_per_tok = 0.0
    for _ in range(max(1, reps)):
        prompt = torch.randint(0, vocab_size, (seq_len,), device=device)
        ms, iters_pt = profile_forward(model, prompt)
        total_ms += ms / max(seq_len, 1)
        total_iters_per_tok += iters_pt
    avg_ms_per_tok = total_ms / max(reps, 1)
    avg_iters_per_tok = total_iters_per_tok / max(reps, 1)
    return avg_ms_per_tok, avg_iters_per_tok


def decode_ms_per_token(model: UELM4, vocab_size: int, prompt_len: int = 64, new_tokens: int = 32, device: torch.device | None = None) -> float:
    device = device or next(model.parameters()).device
    prompt = torch.randint(0, vocab_size, (prompt_len,), device=device)
    return profile_decode(model, prompt, new_tokens=new_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="small")
    parser.add_argument("--corpus", type=Path, default=Path("data/tiny_shakespeare_excerpt.txt"))
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1000, help="Max epochs cap (used as failsafe)")
    parser.add_argument("--seq-len", type=int, default=256, help="Fixed packed sequence length for dataloader")
    parser.add_argument("--tokens", type=int, default=0, help="Target tokens_seen per config; if 0, falls back to epochs cap")
    parser.add_argument("--token-budget", type=int, default=0, help="Deprecated alias for --tokens")
    parser.add_argument("--width", type=int, default=0, help="Optional model width override (d)")
    parser.add_argument("--vocab-size", type=int, default=0, help="Optional vocab size override")
    parser.add_argument("--grid", type=str, default="", help="Optional grid string: T=1,2,3;k=16,32;wmf=0,1;be=1.2,1.5")
    parser.add_argument("--profile", type=str, choices=["fast", "balanced", "quality"], help="Preset: fast(T1,k16,KL,be1.5), balanced(T2,k16,KL,be1.5), quality(T3,k32,KL,be1.5)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--energy-reg", type=float, default=0.0, help="Optional energy regularization weight in training loss (default 0)")
    args = parser.parse_args()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ensure_corpus(args.corpus)
    lines = lines * max(args.repeat, 1)

    device = torch.device(args.device)

    # Parse grid string if provided
    def _parse_grid(s: str):
        if not s:
            return None
        parts = {}
        for seg in s.split(";"):
            if not seg.strip():
                continue
            key, vals = seg.split("=", 1)
            parts[key.strip()] = [v.strip() for v in vals.split(",") if v.strip()]
        T_vals = [int(x) for x in parts.get("T", [])]
        k_vals = [int(x) for x in parts.get("k", [])]
        wmf_vals = [bool(int(x)) for x in parts.get("wmf", [])]
        be_vals = [float(x) for x in parts.get("be", [])]
        if not (T_vals and k_vals and wmf_vals and be_vals):
            return None
        return T_vals, k_vals, wmf_vals, be_vals

    # Apply profile presets if provided (overrides --grid)
    if args.profile:
        if args.profile == "fast":
            T_VALUES, K_VALUES, WMF_VALUES, BETA_END_VALUES = [1], [16], [False], [1.5]
        elif args.profile == "balanced":
            T_VALUES, K_VALUES, WMF_VALUES, BETA_END_VALUES = [2], [16], [False], [1.5]
        elif args.profile == "quality":
            T_VALUES, K_VALUES, WMF_VALUES, BETA_END_VALUES = [3], [32], [False], [1.5]
        else:  # pragma: no cover
            raise SystemExit(f"Unknown profile: {args.profile}")
    else:
        parsed = _parse_grid(args.grid)
        if parsed is None:
            T_VALUES = [1, 2, 3]
            K_VALUES = [16, 32]
            WMF_VALUES = [False, True]
            BETA_END_VALUES = [1.2, 1.5]
        else:
            T_VALUES, K_VALUES, WMF_VALUES, BETA_END_VALUES = parsed

    summary_rows = []
    profiles = []
    histories: dict[str, list[dict[str, float]]] = {}

    for T, k, use_wmf, beta_end in product(T_VALUES, K_VALUES, WMF_VALUES, BETA_END_VALUES):
        label = f"T{T}_k{k}_{'wmf' if use_wmf else 'kl'}_be{beta_end}"
        solver_overrides = {
            "T_train": T,
            "T_infer": T,
            "beta_start": 0.5,
            "beta_end": beta_end,
            "use_wmf": use_wmf,
            "early_exit_tol": 5e-3 if use_wmf else 1e-2,
        }
        if use_wmf:
            solver_overrides.update({
                "wmf_iters": 2,
                "wmf_eps": 0.01,
                "wmf_cost_scale": 0.1,
                "tau_start": 0.1,
                "tau_end": 0.1,
            })

        model_d = args.width if args.width > 0 else (96 if use_wmf else 64)
        model_vocab = args.vocab_size if args.vocab_size > 0 else (512 if use_wmf else 256)
        overrides = {
            "model": {"d": model_d, "vocab_size": model_vocab},
            "memory": {"shortlist_k": k},
            "solver": solver_overrides,
        }

        cfg = load_config(args.config, overrides)
        dataloader = build_dataloader_from_lines(lines, cfg, seq_len=args.seq_len, batch_size=2)
        model = UELM4(cfg).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4)

        # Token budget logic (match tokens_seen across configs)
        batch_size = 2
        num_batches = math.ceil(len(lines) / batch_size)
        tokens_per_epoch = num_batches * batch_size * int(args.seq_len)
        tokens_target = int(args.tokens) if int(args.tokens) > 0 else int(args.token_budget)
        if tokens_target <= 0:
            tokens_target = int(args.epochs) * tokens_per_epoch

        history: list[dict[str, float]] = []
        tokens_seen = 0
        epochs = 0
        start_conf = time.perf_counter()
        while tokens_seen < tokens_target and epochs < int(args.epochs):
            epoch = epochs
            epoch_start = time.perf_counter()
            metrics = train_epoch(model, optimiser, dataloader, device, energy_reg=float(args.energy_reg))
            elapsed_epoch = time.perf_counter() - epoch_start
            history.append({
                "epoch": epoch + 1,
                "loss": metrics["loss"],
                "perplexity": metrics["perplexity"],
                "energy": metrics["energy"],
                "iters_per_token": metrics["iters_per_token"],
                "seconds": elapsed_epoch,
            })
            print(
                f"[{label}] epoch={epoch+1} loss={metrics['loss']:.4f} "
                f"ppl={metrics['perplexity']:.2f} energy={metrics['energy']:.4f} ({elapsed_epoch:.1f}s)"
            )
            epochs += 1
            tokens_seen += tokens_per_epoch
        total_elapsed = time.perf_counter() - start_conf
        histories[label] = history

        # Per-run profiles
        fwd_ms_per_tok, iters_forward = forward_ms_per_token(model, cfg.model.vocab_size, seq_len=256, reps=3, device=device)
        dec_ms_per_tok = decode_ms_per_token(model, cfg.model.vocab_size, prompt_len=64, new_tokens=32, device=device)
        # Energy breakdown on a single forward
        with torch.no_grad():
            prompt_energy = torch.randint(0, cfg.model.vocab_size, (min(256, cfg.model.vocab_size),), device=device)
            _, st_energy, _ = model(prompt_energy, return_state=True)
        memory_table = model.memory.landmarks_view() if hasattr(model.memory, "landmarks_view") else model.memory
        terms = compute_energy_terms(st_energy, memory_table, cfg.solver.rho)
        energy_primal = float(terms["primal"].detach())
        energy_dual = float(terms["dual"].detach())
        energy_entropy = float(terms["entropy"].detach())
        energy_profile_total = float(total_energy(st_energy, memory_table, cfg.solver.rho).detach())

        final = history[-1]
        cfg_dict = asdict(cfg)
        # Hardware info
        dev_type = device.type
        if dev_type == "cuda" and torch.cuda.is_available():
            hw_name = torch.cuda.get_device_name(0)
        else:
            hw_name = platform.processor() or "cpu"
        try:
            p = next(model.parameters())
            if p.dtype == torch.float32:
                dtype_name = "fp32"
            elif p.dtype == torch.bfloat16:
                dtype_name = "bf16"
            elif p.dtype == torch.float16:
                dtype_name = "fp16"
            else:
                dtype_name = str(p.dtype)
        except StopIteration:
            dtype_name = "fp32"
        run_record = {
            "label": label,
            "config": {
                "model": {"d": cfg_dict["model"]["d"], "vocab_size": cfg_dict["model"]["vocab_size"]},
                "solver": {
                    "T_train": cfg_dict["solver"]["T_train"],
                    "T_infer": cfg_dict["solver"]["T_infer"],
                    "beta_end": cfg_dict["solver"]["beta_end"],
                },
                "memory": {"K": cfg_dict["memory"]["K"], "shortlist_k": cfg_dict["memory"]["shortlist_k"]},
            },
            "energy_reg": float(args.energy_reg),
            "final_loss": float(final["loss"]),
            "perplexity": float(final["perplexity"]),
            "final_energy": float(final["energy"]),
            "iters_per_token": float(final["iters_per_token"]),
            "decode_ms_per_token": float(dec_ms_per_tok),
            "forward_ms_per_token": float(fwd_ms_per_tok),
            "total_seconds": float(total_elapsed),
            "tokens_seen": int(tokens_seen),
            "epochs": int(epochs),
            "hardware": {"device": dev_type, "name": hw_name, "dtype": dtype_name},
            "energy_primal": energy_primal,
            "energy_dual": energy_dual,
            "energy_entropy": energy_entropy,
            "energy_profile_total": energy_profile_total,
        }
        # Append to global results list
        try:
            results.append(run_record)
        except NameError:
            # Back-compat in case results wasn't defined
            results = [run_record]

    (out_dir / "grid_results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "history.json").write_text(json.dumps(histories, indent=2))

    with (out_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = [
            "label",
            "final_loss",
            "perplexity",
            "final_energy",
            "iters_per_token",
            "decode_ms_per_token",
            "forward_ms_per_token",
            "total_seconds",
            "tokens_seen",
            "epochs",
            "T",
            "k",
            "use_wmf",
            "beta_end",
            "device",
            "hardware_name",
            "dtype",
        ]
        writer.writerow(header)
        for row in results:
            solver = row["config"]["solver"]
            mem = row["config"]["memory"]
            writer.writerow([
                row["label"],
                row["final_loss"],
                row["perplexity"],
                row["final_energy"],
                row["iters_per_token"],
                row["decode_ms_per_token"],
                row["forward_ms_per_token"],
                row["total_seconds"],
                row["tokens_seen"],
                row["epochs"],
                solver["T_train"],
                mem["shortlist_k"],
                int(solver.get("use_wmf", 0)),
                solver["beta_end"],
                row.get("hardware", {}).get("device", ""),
                row.get("hardware", {}).get("name", ""),
                row.get("hardware", {}).get("dtype", ""),
            ])

    try:
        import matplotlib.pyplot as plt

        for row in results:
            label = row["label"]
            hist = histories[label]
            epochs = [item["epoch"] for item in hist]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epochs, [item["loss"] for item in hist], marker="o")
            ax.set_title(f"Loss – {label}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"loss_{label}.png")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epochs, [item["energy"] for item in hist], marker="o", color="#F58518")
            ax.set_title(f"Energy – {label}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Energy")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"energy_{label}.png")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epochs, [item["iters_per_token"] for item in hist], marker="o", color="#54A24B")
            ax.set_title(f"Iterations/token – {label}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("iters/token")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"iters_{label}.png")
    except Exception as exc:  # pragma: no cover
        (out_dir / "plot_warning.txt").write_text(f"Plot generation skipped: {exc}\n")

    # Build Pareto and write report
    def _pareto_frontier(points: list[dict]) -> list[dict]:
        pts = sorted(points, key=lambda r: (r["perplexity"], r["decode_ms_per_token"]))
        frontier: list[dict] = []
        best_decode = float("inf")
        for r in pts:
            if r["decode_ms_per_token"] < best_decode:
                frontier.append(r)
                best_decode = r["decode_ms_per_token"]
        return frontier

    best_ppl = min(results, key=lambda r: r["perplexity"]) if results else None
    best_speed = min(results, key=lambda r: r["decode_ms_per_token"]) if results else None
    best_eff = min(results, key=lambda r: r["perplexity"] * r["decode_ms_per_token"]) if results else None

    pareto = _pareto_frontier(results)
    with (out_dir / "pareto.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "ppl", "decode_ms_per_token", "T", "k", "use_wmf", "beta_end"])
        for r in pareto:
            s = r["config"]["solver"]
            m = r["config"]["memory"]
            writer.writerow([r["label"], r["perplexity"], r["decode_ms_per_token"], s["T_train"], m["shortlist_k"], int(s.get("use_wmf", 0)), s["beta_end"]])

    report_path = out_dir / "report.md"
    with report_path.open("w") as fh:
        fh.write("# Phase-B Grid Sweep Report\n\n")
        fh.write(f"Corpus: {args.corpus}\n\n")
        fh.write("## Mini Summary\n\n")
        if best_ppl:
            fh.write(f"- Best by ppl: {best_ppl['label']} (ppl={best_ppl['perplexity']:.3f}, ms/tok={best_ppl['decode_ms_per_token']:.2f})\n")
        if best_speed:
            fh.write(f"- Best by speed: {best_speed['label']} (ms/tok={best_speed['decode_ms_per_token']:.2f}, ppl={best_speed['perplexity']:.3f})\n")
        if best_eff:
            fh.write(f"- Best efficiency: {best_eff['label']} (ppl*ms/tok={(best_eff['perplexity']*best_eff['decode_ms_per_token']):.3f})\n")
        fh.write("\n")
        fh.write("## Summary\n\n")
        fh.write("| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |\n")
        fh.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in results:
            s = row["config"]["solver"]
            m = row["config"]["memory"]
            fh.write(
                f"| {row['label']} | {row['perplexity']:.3f} | {row['decode_ms_per_token']:.2f} | {row['forward_ms_per_token']:.2f} | "
                f"{row['iters_per_token']:.3f} | {row['epochs']} | {row['tokens_seen']} | {row['total_seconds']:.1f} | "
                f"{s['T_train']} | {m['shortlist_k']} | {int(s.get('use_wmf', 0))} | {s['beta_end']} |\n"
            )

        fh.write("\n## Pareto (ppl vs decode ms/token)\n\n")
        for r in pareto:
            s = r["config"]["solver"]
            m = r["config"]["memory"]
            fh.write(
                f"- {r['label']}: ppl={r['perplexity']:.3f}, ms/token={r['decode_ms_per_token']:.2f} (T={s['T_train']}, k={m['shortlist_k']}, wmf={int(s.get('use_wmf', 0))}, be={s['beta_end']})\n"
            )

        fh.write("\n## Energy Breakdown (profile prompt)\n\n")
        for row in results:
            fh.write(
                f"- {row['label']}: primal={row.get('energy_primal', float('nan')):.4f}, dual={row.get('energy_dual', float('nan')):.4f}, entropy={row.get('energy_entropy', float('nan')):.4f}, total={row.get('energy_profile_total', float('nan')):.4f}\n"
            )

        fh.write("\n## Plots\n")
        for row in results:
            label = row["label"]
            for prefix in ("loss", "energy", "iters"):
                png = Path(f"{prefix}_{label}.png")
                if png.exists():
                    fh.write(f"\n![{prefix} {label}]({png.name})\n")

    if best_ppl:
        print(f"best by ppl: {best_ppl['label']} ppl={best_ppl['perplexity']:.3f} ms/tok={best_ppl['decode_ms_per_token']:.2f}")
    if best_speed:
        print(f"best by speed: {best_speed['label']} ms/tok={best_speed['decode_ms_per_token']:.2f} ppl={best_speed['perplexity']:.3f}")
    if best_eff:
        print(f"best efficiency: {best_eff['label']} ppl*ms/tok={(best_eff['perplexity']*best_eff['decode_ms_per_token']):.3f}")
    print(f"Grid sweep complete. Report: {report_path}")
    return


if __name__ == "__main__":
    main()
