# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T2_k16_kl_be1.5 (ppl=1.140, ms/tok=4.25)
- Best by speed: T2_k16_kl_be1.5 (ms/tok=4.25, ppl=1.140)
- Best efficiency: T2_k16_kl_be1.5 (ppl*ms/tok=4.848)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2_k16_kl_be1.5 | 1.140 | 4.25 | 0.03 | 0.000 | 5 | 537600 | 51.8 | 2 | 16 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T2_k16_kl_be1.5: ppl=1.140, ms/token=4.25 (T=2, k=16, wmf=0, be=1.5)

## Plots
