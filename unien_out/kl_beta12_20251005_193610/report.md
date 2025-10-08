# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T2_k16_kl_be1.6 (ppl=1.036, ms/tok=5.66)
- Best by speed: T2_k16_kl_be1.3 (ms/tok=2.91, ppl=1.048)
- Best efficiency: T2_k16_kl_be1.3 (ppl*ms/tok=3.052)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k16_kl_be1.3 | 1.047 | 4.19 | 0.02 | 0.000 | 10 | 2150400 | 128.9 | 1 | 16 | 0 | 1.3 |
| T1_k16_kl_be1.4 | 1.053 | 3.60 | 0.03 | 0.000 | 10 | 2150400 | 125.4 | 1 | 16 | 0 | 1.4 |
| T1_k16_kl_be1.5 | 1.053 | 4.22 | 0.03 | 0.000 | 10 | 2150400 | 124.9 | 1 | 16 | 0 | 1.5 |
| T1_k16_kl_be1.6 | 1.052 | 3.68 | 0.03 | 0.000 | 10 | 2150400 | 125.2 | 1 | 16 | 0 | 1.6 |
| T2_k16_kl_be1.3 | 1.048 | 2.91 | 0.02 | 0.000 | 10 | 2150400 | 162.9 | 2 | 16 | 0 | 1.3 |
| T2_k16_kl_be1.4 | 1.044 | 6.27 | 0.05 | 0.000 | 10 | 2150400 | 159.5 | 2 | 16 | 0 | 1.4 |
| T2_k16_kl_be1.5 | 1.037 | 6.99 | 0.04 | 0.000 | 10 | 2150400 | 174.2 | 2 | 16 | 0 | 1.5 |
| T2_k16_kl_be1.6 | 1.036 | 5.66 | 0.04 | 0.000 | 10 | 2150400 | 173.0 | 2 | 16 | 0 | 1.6 |

## Pareto (ppl vs decode ms/token)

- T2_k16_kl_be1.6: ppl=1.036, ms/token=5.66 (T=2, k=16, wmf=0, be=1.6)
- T1_k16_kl_be1.3: ppl=1.047, ms/token=4.19 (T=1, k=16, wmf=0, be=1.3)
- T2_k16_kl_be1.3: ppl=1.048, ms/token=2.91 (T=2, k=16, wmf=0, be=1.3)

## Plots
