# attention shape boundaries

Coverage and finite timing/correctness checks passed. Full machine-readable table: shape_comparison.csv.

Logical load counts are valid-element requests from the source algorithm, not measured DRAM traffic. Output slot utilization includes tile tails. Register blocking changes both per-thread work and block tile. These are directed validation shapes, not an unseen independent test set.

| Case | Coverage | Fastest warp configuration | Paired speedup vs w4 |
|---|---|---|---:|
| tiny | diagnostic | w8 | 1.0879x |
| odd_gqa | diagnostic | w16 | 1.2177x |
| peaky | diagnostic | w8 | 1.1168x |
| very_peaky | diagnostic | w8 | 1.3276x |
| late_max | diagnostic | w16 | 1.2674x |
| negative_scores | diagnostic | w8 | 1.1151x |
| uniform | diagnostic | w16 | 1.3126x |
| causal | diagnostic | w16 | 1.3347x |
| cache_q5 | diagnostic | w8 | 1.1177x |
| cache_q1 | diagnostic | w16 | 1.4356x |
| masked_prefix | diagnostic | w8 | 1.0943x |
| dim96 | diagnostic | w8 | 1.1832x |
| decoder | model_representative | w16 | 1.5982x |
| encoder | model_representative | w16 | 1.3213x |
| length_32 | diagnostic | w8 | 1.2689x |
| length_128 | diagnostic | w16 | 1.3486x |
| length_512 | diagnostic | w16 | 1.3338x |
| causal_tail65 | diagnostic | w8 | 1.3796x |

Length/GQA/causal conditions are compared separately; stress cases do not count as additional model workloads. This table tests whether the long-encoder choice transfers to short or low-parallelism inputs.
