# Phase-B Grid Sweep Report

Corpus: /Users/trevormaxwell/Desktop/unien/data/corpus_wikitext2.txt

## Mini Summary

- Best by ppl: T2_k16_kl_be1.4 (ppl=1.307, ms/tok=3.98)
- Best by speed: T2_k16_kl_be1.4 (ms/tok=3.98, ppl=1.307)
- Best efficiency: T2_k16_kl_be1.4 (ppl*ms/tok=5.198)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T2_k16_kl_be1.4 | 1.307 | 3.98 | 0.04 | 0.000 | 49 | 2007040 | 157.3 | 2 | 16 | 0 | 1.4 |

## Pareto (ppl vs decode ms/token)

- T2_k16_kl_be1.4: ppl=1.307, ms/token=3.98 (T=2, k=16, wmf=0, be=1.4)

## Energy Breakdown (profile prompt)

- T2_k16_kl_be1.4: primal=175.0296, dual=182.7976, entropy=0.5685, total=357.8329

## Plots
