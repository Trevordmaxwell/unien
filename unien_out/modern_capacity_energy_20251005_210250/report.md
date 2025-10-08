# Phase-B Grid Sweep Report

Corpus: /Users/trevormaxwell/Desktop/unien/data/corpus_wikitext2.txt

## Mini Summary

- Best by ppl: T1_k16_kl_be1.5 (ppl=1.002, ms/tok=3.76)
- Best by speed: T1_k16_kl_be1.5 (ms/tok=3.76, ppl=1.002)
- Best efficiency: T1_k16_kl_be1.5 (ppl*ms/tok=3.768)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k16_kl_be1.5 | 1.002 | 3.76 | 0.04 | 0.000 | 49 | 2007040 | 165.4 | 1 | 16 | 0 | 1.5 |
| T2_k16_kl_be1.5 | 1.067 | 5.85 | 0.04 | 0.000 | 49 | 2007040 | 226.2 | 2 | 16 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T1_k16_kl_be1.5: ppl=1.002, ms/token=3.76 (T=1, k=16, wmf=0, be=1.5)

## Energy Breakdown (profile prompt)

- T1_k16_kl_be1.5: primal=11.8871, dual=11.8871, entropy=1.0585, total=23.7849
- T2_k16_kl_be1.5: primal=99.0260, dual=105.2321, entropy=0.6751, total=204.2649

## Plots
