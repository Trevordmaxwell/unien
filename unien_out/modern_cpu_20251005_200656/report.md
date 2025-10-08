# Phase-B Grid Sweep Report

Corpus: /Users/trevormaxwell/Desktop/unien/data/corpus_wikitext2.txt

## Mini Summary

- Best by ppl: T3_k32_kl_be1.5 (ppl=1.004, ms/tok=14.03)
- Best by speed: T1_k16_kl_be1.5 (ms/tok=4.26, ppl=1.010)
- Best efficiency: T1_k16_kl_be1.5 (ppl*ms/tok=4.303)

## Summary

| label | ppl | decode ms/token | forward ms/token | iters/token | epochs | tokens | total s | T | k | wmf | beta_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T1_k16_kl_be1.5 | 1.010 | 4.26 | 0.02 | 0.000 | 49 | 2007040 | 105.1 | 1 | 16 | 0 | 1.5 |
| T1_k32_kl_be1.5 | 1.006 | 4.62 | 0.03 | 0.000 | 49 | 2007040 | 139.3 | 1 | 32 | 0 | 1.5 |
| T2_k16_kl_be1.5 | 1.006 | 4.90 | 0.13 | 0.000 | 49 | 2007040 | 164.3 | 2 | 16 | 0 | 1.5 |
| T2_k32_kl_be1.5 | 1.013 | 8.20 | 0.08 | 0.000 | 49 | 2007040 | 196.0 | 2 | 32 | 0 | 1.5 |
| T3_k16_kl_be1.5 | 1.005 | 10.17 | 0.09 | 0.000 | 49 | 2007040 | 210.4 | 3 | 16 | 0 | 1.5 |
| T3_k32_kl_be1.5 | 1.004 | 14.03 | 0.11 | 0.000 | 49 | 2007040 | 251.2 | 3 | 32 | 0 | 1.5 |

## Pareto (ppl vs decode ms/token)

- T3_k32_kl_be1.5: ppl=1.004, ms/token=14.03 (T=3, k=32, wmf=0, be=1.5)
- T3_k16_kl_be1.5: ppl=1.005, ms/token=10.17 (T=3, k=16, wmf=0, be=1.5)
- T1_k32_kl_be1.5: ppl=1.006, ms/token=4.62 (T=1, k=32, wmf=0, be=1.5)
- T1_k16_kl_be1.5: ppl=1.010, ms/token=4.26 (T=1, k=16, wmf=0, be=1.5)

## Energy Breakdown (profile prompt)

- T1_k16_kl_be1.5: primal=1366.3940, dual=1366.3940, entropy=0.0728, total=2732.7888
- T1_k32_kl_be1.5: primal=5311.3774, dual=5311.3774, entropy=0.0063, total=10622.7549
- T2_k16_kl_be1.5: primal=27525.9102, dual=28402.0566, entropy=0.0110, total=55927.9688
- T2_k32_kl_be1.5: primal=14379.2168, dual=14771.1133, entropy=0.0207, total=29150.3301
- T3_k16_kl_be1.5: primal=34662.1289, dual=37642.8672, entropy=0.0075, total=72305.0000
- T3_k32_kl_be1.5: primal=22778.1738, dual=24652.9238, entropy=0.0149, total=47431.0977

## Plots
