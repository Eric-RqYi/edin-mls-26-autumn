# Reuse validation

No samples removed. See history_manifest.json for existing evidence coverage.

# gemm shape boundaries

Coverage and finite timing/correctness checks passed. Full machine-readable table: shape_comparison.csv.

Logical load counts are valid-element requests from the source algorithm, not measured DRAM traffic. Output slot utilization includes tile tails. Register blocking changes both per-thread work and block tile. These are directed validation shapes, not an unseen independent test set.

Against tiled16, rb4x4 improves 20/37 directed cases and regresses on 17. This count is not weighted by model call frequency.

| Case | M,N,K | rb4x4 paired speedup | Valid output slots |
|---|---|---:|---:|
| n_scan_m32_n256 | 32,256,2048 | 0.2923x | 0.500 |
| edge_gelu | 45,77,129 | 0.3178x | 0.423 |
| edge_odd | 37,100,33 | 0.4549x | 0.452 |
| encoder_q | 1500,1280,1280 | 2.7108x | 0.977 |
| encoder_fc1 | 1500,5120,1280 | 2.6986x | 0.977 |
| encoder_fc2 | 1500,1280,5120 | 2.6890x | 0.977 |

Use M=31/32/33 and 63/64/65 rows to check discontinuities. The slot-fraction and request-count columns quantify tile waste and logical reuse. Narrow-N and adjacent N/K rows test the fixed dispatch boundary; they do not modify it. Actual GPU traffic, occupancy and cache behavior require separate measurements.


# m1 shape boundaries

Coverage and finite timing/correctness checks passed. Full machine-readable table: shape_comparison.csv.

Logical load counts are valid-element requests from the source algorithm, not measured DRAM traffic. Output slot utilization includes tile tails. Register blocking changes both per-thread work and block tile. These are directed validation shapes, not an unseen independent test set.

The first two configurations explicitly disable the M1 path; the third enables it.


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


# ASR controlled comparison

| Configuration | Median ms | Mean ms | IQR ms | Paired speedup |
|---|---:|---:|---:|---:|
| base | 3372.003 | 3371.929 | 1.774 | 1.00000x |
| hybrid | 3336.703 | 3336.636 | 1.579 | 1.01060x |
| attention | 3129.042 | 3129.095 | 1.061 | 1.07756x |
| combined | 3093.894 | 3093.976 | 0.997 | 1.08985x |
| torch | 1330.317 | 1330.451 | 1.512 | 2.53449x |

Attention factor varies only block query/warp count (4/16). Hybrid factor varies only the fixed narrow-N dispatch. Every timed token sequence matches the reference. All samples retained; one supplied audio, three processes in one allocation. Profiles are separate diagnostics and include original ops allocation/launch boundaries.
