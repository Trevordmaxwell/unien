# Phase-B Grid Sweep Report

Corpus: /Users/trevormaxwell/Desktop/unien/data/corpus_wikitext2.txt

## Mini Summary

- Best by ppl: T3_k32_kl_be1.5 (ppl=13.032, ms/tok=3.99)
- Best by speed: T3_k32_kl_be1.5 (ms/tok=3.99, ppl=13.032)
- Best efficiency: T3_k32_kl_be1.5 (ppl*ms/tok=52.057)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T3_k32_kl_be1.5 | 13.032 | 3.99 | 0.04 | 0.000 | 49 | 2007040 | 120.1 | 3 | 32 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T3_k32_kl_be1.5: ppl=13.032, ms/token=3.99 (T=3, k=32, wmf=0, be=1.5)

## Energy Breakdown (profile prompt)

- T3_k32_kl_be1.5: primal=1800.3101, dual=2023.5468, entropy=0.2859, total=3823.8599

## Plots
