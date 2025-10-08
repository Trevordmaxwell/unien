# Phase-B Grid Sweep Report

Corpus: /Users/trevormaxwell/Desktop/unien/data/corpus_wikitext2.txt

## Mini Summary

- Best by ppl: T1_k32_kl_be1.5 (ppl=1.104, ms/tok=3.61)
- Best by speed: T1_k32_kl_be1.5 (ms/tok=3.61, ppl=1.104)
- Best efficiency: T1_k32_kl_be1.5 (ppl*ms/tok=3.980)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k32_kl_be1.5 | 1.104 | 3.61 | 0.03 | 0.000 | 489 | 20029440 | 1625.4 | 1 | 32 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T1_k32_kl_be1.5: ppl=1.104, ms/token=3.61 (T=1, k=32, wmf=0, be=1.5)

## Energy Breakdown (profile prompt)

- T1_k32_kl_be1.5: primal=29.2369, dual=29.2369, entropy=0.7836, total=58.4816

## Plots
