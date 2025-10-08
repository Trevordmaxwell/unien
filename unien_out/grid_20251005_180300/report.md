# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T2_k16_kl_be1.5 (ppl=1.132, ms/tok=2.69)
- Best by speed: T1_k16_kl_be1.2 (ms/tok=1.61, ppl=1.147)
- Best efficiency: T1_k16_kl_be1.2 (ppl*ms/tok=1.853)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k16_kl_be1.2 | 1.147 | 1.61 | 0.01 | 0.004 | 5 | 537600 | 23.9 | 1 | 16 | 1.2 |
| T1_k16_kl_be1.5 | 1.153 | 1.62 | 0.01 | 0.004 | 5 | 537600 | 24.1 | 1 | 16 | 1.5 |
| T1_k32_kl_be1.2 | 1.145 | 1.93 | 0.02 | 0.004 | 5 | 537600 | 27.2 | 1 | 32 | 1.2 |
| T1_k32_kl_be1.5 | 1.151 | 1.83 | 0.02 | 0.004 | 5 | 537600 | 27.2 | 1 | 32 | 1.5 |
| T2_k16_kl_be1.2 | 1.133 | 2.54 | 0.02 | 0.008 | 5 | 537600 | 30.0 | 2 | 16 | 1.2 |
| T2_k16_kl_be1.5 | 1.132 | 2.69 | 0.02 | 0.008 | 5 | 537600 | 30.1 | 2 | 16 | 1.5 |
| T2_k32_kl_be1.2 | 1.136 | 2.96 | 0.03 | 0.008 | 5 | 537600 | 34.6 | 2 | 32 | 1.2 |
| T2_k32_kl_be1.5 | 1.133 | 3.13 | 0.03 | 0.008 | 5 | 537600 | 35.5 | 2 | 32 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T2_k16_kl_be1.5: ppl=1.132, ms/token=2.69 (T=2, k=16, be=1.5)
- T2_k16_kl_be1.2: ppl=1.133, ms/token=2.54 (T=2, k=16, be=1.2)
- T1_k32_kl_be1.2: ppl=1.145, ms/token=1.93 (T=1, k=32, be=1.2)
- T1_k16_kl_be1.2: ppl=1.147, ms/token=1.61 (T=1, k=16, be=1.2)

## Plots
