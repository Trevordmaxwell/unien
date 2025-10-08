# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T3_k32_wmf_be1.5 (ppl=1.007, ms/tok=25.52)
- Best by speed: T2_k32_wmf_be1.5 (ms/tok=18.47, ppl=1.016)
- Best efficiency: T2_k32_wmf_be1.5 (ppl*ms/tok=18.763)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2_k32_wmf_be1.2 | 1.009 | 22.90 | 0.20 | 0.000 | 10 | 2150400 | 586.6 | 2 | 32 | 0 | 1.2 |
| T2_k32_wmf_be1.5 | 1.016 | 18.47 | 0.20 | 0.000 | 10 | 2150400 | 580.9 | 2 | 32 | 0 | 1.5 |
| T3_k32_wmf_be1.2 | 1.058 | 23.23 | 0.24 | 0.000 | 10 | 2150400 | 628.5 | 3 | 32 | 0 | 1.2 |
| T3_k32_wmf_be1.5 | 1.007 | 25.52 | 0.27 | 0.000 | 10 | 2150400 | 785.5 | 3 | 32 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T3_k32_wmf_be1.5: ppl=1.007, ms/token=25.52 (T=3, k=32, wmf=0, be=1.5)
- T2_k32_wmf_be1.2: ppl=1.009, ms/token=22.90 (T=2, k=32, wmf=0, be=1.2)
- T2_k32_wmf_be1.5: ppl=1.016, ms/token=18.47 (T=2, k=32, wmf=0, be=1.5)

## Plots
