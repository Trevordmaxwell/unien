# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T3_k32_kl_be1.6 (ppl=1.013, ms/tok=5.46)
- Best by speed: T3_k32_kl_be1.6 (ms/tok=5.46, ppl=1.013)
- Best efficiency: T3_k32_kl_be1.6 (ppl*ms/tok=5.536)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T3_k32_kl_be1.5 | 1.036 | 11.23 | 0.11 | 0.000 | 10 | 2150400 | 260.0 | 3 | 32 | 0 | 1.5 |
| T3_k32_kl_be1.6 | 1.013 | 5.46 | 0.06 | 0.000 | 10 | 2150400 | 201.9 | 3 | 32 | 0 | 1.6 |

## Pareto (ppl vs decode ms/token)

- T3_k32_kl_be1.6: ppl=1.013, ms/token=5.46 (T=3, k=32, wmf=0, be=1.6)

## Energy Breakdown (profile prompt)

- T3_k32_kl_be1.5: primal=1559.2268, dual=1791.5178, entropy=0.0921, total=3350.7456
- T3_k32_kl_be1.6: primal=33010.4609, dual=38015.2031, entropy=0.2064, total=71025.6641

## Plots
