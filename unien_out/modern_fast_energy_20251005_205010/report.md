# Phase-B Grid Sweep Report

Corpus: /Users/trevormaxwell/Desktop/unien/data/corpus_wikitext2.txt

## Mini Summary

- Best by ppl: T1_k16_kl_be1.5 (ppl=10.506, ms/tok=1.61)
- Best by speed: T1_k16_kl_be1.5 (ms/tok=1.61, ppl=10.506)
- Best efficiency: T1_k16_kl_be1.5 (ppl*ms/tok=16.906)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k16_kl_be1.5 | 10.506 | 1.61 | 0.01 | 0.000 | 49 | 2007040 | 62.6 | 1 | 16 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T1_k16_kl_be1.5: ppl=10.506, ms/token=1.61 (T=1, k=16, wmf=0, be=1.5)

## Energy Breakdown (profile prompt)

- T1_k16_kl_be1.5: primal=1788.7528, dual=1788.7528, entropy=0.0128, total=3577.5059

## Plots
