# Phase-B Grid Sweep Report

Corpus: data/tiny_shakespeare_excerpt.txt

## Mini Summary

- Best by ppl: T1_k16_kl_be1.5 (ppl=1.129, ms/tok=3.69)
- Best by speed: T1_k16_kl_be1.5 (ms/tok=3.69, ppl=1.129)
- Best efficiency: T1_k16_kl_be1.5 (ppl*ms/tok=4.165)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k16_kl_be1.5 | 1.129 | 3.69 | 0.02 | 0.000 | 5 | 537600 | 40.4 | 1 | 16 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T1_k16_kl_be1.5: ppl=1.129, ms/token=3.69 (T=1, k=16, wmf=0, be=1.5)

## Plots
