# Phase-B Grid Sweep Report

Corpus: /Users/trevormaxwell/Desktop/unien/data/corpus_wikitext2.txt

## Mini Summary

- Best by ppl: T3_k32_wmf_be1.5 (ppl=1.000, ms/tok=20.99)
- Best by speed: T3_k32_wmf_be1.5 (ms/tok=20.99, ppl=1.000)
- Best efficiency: T3_k32_wmf_be1.5 (ppl*ms/tok=20.994)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T3_k32_wmf_be1.2 | 1.000 | 29.80 | 0.29 | 0.000 | 49 | 2007040 | 757.2 | 3 | 32 | 0 | 1.2 |
| T3_k32_wmf_be1.5 | 1.000 | 20.99 | 0.23 | 0.000 | 49 | 2007040 | 661.9 | 3 | 32 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T3_k32_wmf_be1.5: ppl=1.000, ms/token=20.99 (T=3, k=32, wmf=0, be=1.5)

## Energy Breakdown (profile prompt)

- T3_k32_wmf_be1.2: primal=7280.0757, dual=7860.0571, entropy=1.4083, total=15140.1465
- T3_k32_wmf_be1.5: primal=6539.7065, dual=6898.7769, entropy=1.3928, total=13438.4971

## Plots
