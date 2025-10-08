# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T1_k16_kl_be1.5 (ppl=1.203, ms/tok=2.54)
- Best by speed: T1_k16_kl_be1.5 (ms/tok=2.54, ppl=1.203)
- Best efficiency: T1_k16_kl_be1.5 (ppl*ms/tok=3.058)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k16_kl_be1.5 | 1.203 | 2.54 | 0.02 | 0.000 | 5 | 537600 | 33.0 | 1 | 16 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T1_k16_kl_be1.5: ppl=1.203, ms/token=2.54 (T=1, k=16, wmf=0, be=1.5)

## Energy Breakdown (profile prompt)

- T1_k16_kl_be1.5: primal=80.1848, dual=80.1848, entropy=0.4748, total=160.3744

## Plots
