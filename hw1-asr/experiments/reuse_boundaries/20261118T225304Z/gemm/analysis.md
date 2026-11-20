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
