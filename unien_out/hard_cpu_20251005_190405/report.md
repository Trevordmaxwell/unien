# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T3_k32_kl_be1.5 (ppl=1.021, ms/tok=14.33)
- Best by speed: T1_k16_kl_be1.5 (ms/tok=5.56, ppl=1.041)
- Best efficiency: T1_k16_kl_be1.5 (ppl*ms/tok=5.792)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k16_kl_be1.5 | 1.041 | 5.56 | 0.04 | 0.000 | 10 | 2150400 | 121.3 | 1 | 16 | 0 | 1.5 |
| T1_k32_kl_be1.5 | 1.049 | 5.71 | 0.06 | 0.000 | 10 | 2150400 | 157.9 | 1 | 32 | 0 | 1.5 |
| T2_k16_kl_be1.5 | 1.040 | 7.96 | 0.07 | 0.000 | 10 | 2150400 | 185.3 | 2 | 16 | 0 | 1.5 |
| T2_k32_kl_be1.5 | 1.034 | 10.26 | 0.11 | 0.000 | 10 | 2150400 | 214.8 | 2 | 32 | 0 | 1.5 |
| T3_k16_kl_be1.5 | 1.026 | 10.13 | 0.10 | 0.000 | 10 | 2150400 | 233.6 | 3 | 16 | 0 | 1.5 |
| T3_k32_kl_be1.5 | 1.021 | 14.33 | 0.13 | 0.000 | 10 | 2150400 | 272.8 | 3 | 32 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T3_k32_kl_be1.5: ppl=1.021, ms/token=14.33 (T=3, k=32, wmf=0, be=1.5)
- T3_k16_kl_be1.5: ppl=1.026, ms/token=10.13 (T=3, k=16, wmf=0, be=1.5)
- T2_k16_kl_be1.5: ppl=1.040, ms/token=7.96 (T=2, k=16, wmf=0, be=1.5)
- T1_k16_kl_be1.5: ppl=1.041, ms/token=5.56 (T=1, k=16, wmf=0, be=1.5)

## Plots
