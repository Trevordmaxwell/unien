# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T1_k16_kl_be1.5 (ppl=1.250, ms/tok=1.59)
- Best by speed: T1_k16_kl_be1.5 (ms/tok=1.59, ppl=1.250)
- Best efficiency: T1_k16_kl_be1.5 (ppl*ms/tok=1.993)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k16_kl_be1.5 | 1.250 | 1.59 | 0.01 | 0.000 | 5 | 537600 | 23.2 | 1 | 16 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T1_k16_kl_be1.5: ppl=1.250, ms/token=1.59 (T=1, k=16, wmf=0, be=1.5)

## Energy Breakdown (profile prompt)

- T1_k16_kl_be1.5: primal=56.7068, dual=56.7068, entropy=0.4690, total=113.4182

## Plots
