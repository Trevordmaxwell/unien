# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T2_k16_kl_be1.5 (ppl=3.875, ms/tok=2.88)
- Best by speed: T2_k16_kl_be1.5 (ms/tok=2.88, ppl=3.875)
- Best efficiency: T2_k16_kl_be1.5 (ppl*ms/tok=11.147)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2_k16_kl_be1.5 | 3.875 | 2.88 | 0.02 | 0.000 | 5 | 537600 | 30.3 | 2 | 16 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T2_k16_kl_be1.5: ppl=3.875, ms/token=2.88 (T=2, k=16, wmf=0, be=1.5)

## Energy Breakdown (profile prompt)

- T2_k16_kl_be1.5: primal=73.5612, dual=77.9390, entropy=1.3159, total=151.5134

## Plots
