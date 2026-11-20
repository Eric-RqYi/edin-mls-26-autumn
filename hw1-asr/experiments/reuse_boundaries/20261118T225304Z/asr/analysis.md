# ASR controlled comparison

| Configuration | Median ms | Mean ms | IQR ms | Paired speedup |
|---|---:|---:|---:|---:|
| base | 3372.003 | 3371.929 | 1.774 | 1.00000x |
| hybrid | 3336.703 | 3336.636 | 1.579 | 1.01060x |
| attention | 3129.042 | 3129.095 | 1.061 | 1.07756x |
| combined | 3093.894 | 3093.976 | 0.997 | 1.08985x |
| torch | 1330.317 | 1330.451 | 1.512 | 2.53449x |

Attention factor varies only block query/warp count (4/16). Hybrid factor varies only the fixed narrow-N dispatch. Every timed token sequence matches the reference. All samples retained; one supplied audio, three processes in one allocation. Profiles are separate diagnostics and include original ops allocation/launch boundaries.
