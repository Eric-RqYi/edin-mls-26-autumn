# m1 shape boundaries

Coverage and finite timing/correctness checks passed. Full machine-readable table: shape_comparison.csv.

Logical load counts are valid-element requests from the source algorithm, not measured DRAM traffic. Output slot utilization includes tile tails. Register blocking changes both per-thread work and block tile. These are directed validation shapes, not an unseen independent test set.

The first two configurations explicitly disable the M1 path; the third enables it.
