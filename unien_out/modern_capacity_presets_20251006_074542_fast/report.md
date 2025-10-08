# Phase-B Grid Sweep Report

Corpus: /Users/trevormaxwell/Desktop/unien/data/corpus_wikitext2.txt

## Mini Summary

- Best by ppl: T1_k32_kl_be1.5 (ppl=1.006, ms/tok=3.01)
- Best by speed: T1_k32_kl_be1.5 (ms/tok=3.01, ppl=1.006)
- Best efficiency: T1_k32_kl_be1.5 (ppl*ms/tok=3.029)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k32_kl_be1.5 | 1.006 | 3.01 | 0.03 | 0.000 | 49 | 2007040 | 147.6 | 1 | 32 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T1_k32_kl_be1.5: ppl=1.006, ms/token=3.01 (T=1, k=32, wmf=0, be=1.5)

## Energy Breakdown (profile prompt)

- T1_k32_kl_be1.5: primal=22.8917, dual=22.8917, entropy=0.9090, total=45.7925

## Plots
