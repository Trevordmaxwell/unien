# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T3_k32_kl_be1.5 (ppl=1.117, ms/tok=4.41)
- Best by speed: T3_k32_kl_be1.5 (ms/tok=4.41, ppl=1.117)
- Best efficiency: T3_k32_kl_be1.5 (ppl*ms/tok=4.928)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T3_k32_kl_be1.5 | 1.117 | 4.41 | 0.04 | 0.000 | 5 | 537600 | 64.7 | 3 | 32 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T3_k32_kl_be1.5: ppl=1.117, ms/token=4.41 (T=3, k=32, wmf=0, be=1.5)

## Plots
