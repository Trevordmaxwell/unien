# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T1_k16_kl_be1.5 (ppl=1.205, ms/tok=1.51)
- Best by speed: T1_k16_kl_be1.5 (ms/tok=1.51, ppl=1.205)
- Best efficiency: T1_k16_kl_be1.5 (ppl*ms/tok=1.821)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k16_kl_be1.5 | 1.205 | 1.51 | 0.01 | 0.000 | 5 | 537600 | 23.3 | 1 | 16 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T1_k16_kl_be1.5: ppl=1.205, ms/token=1.51 (T=1, k=16, wmf=0, be=1.5)

## Energy Breakdown (profile prompt)

- T1_k16_kl_be1.5: primal=56.6228, dual=56.6228, entropy=1.0534, total=113.2561

## Plots
