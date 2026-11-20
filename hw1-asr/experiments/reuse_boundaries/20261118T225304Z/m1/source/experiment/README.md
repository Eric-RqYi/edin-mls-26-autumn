# Task A: reuse and execution-granularity boundaries

Branch `asr/task-a-boundaries`. The common kernel rollback follows the user's newer instruction over TASK.md's earlier preservation rule. Frozen historical sources and measurements are never modified. All new configurations explicitly set all kernel factors; report text concerns warp count, reuse, and shape boundaries.

From **hw1-asr** on the cluster:

```bash
sbatch --gres=gpu:h200_1g.18gb:1 experiments/reuse_boundaries/run_cluster.sbatch --stage all
```

`all` runs GEMM boundaries, independent M1 builds, attention lengths, then the four-group ASR experiment. Each part gets its own frozen source, SHA256 manifest, independent libraries, unchanged coursework tests, CUDA compiler/host preflight, and non-default-stream reference benchmarks. The dedicated hybrid M=2 sanitizer commands omit `--small` and must finish before model runs. Every sanitizer is mandatory. Default warmup/iterations/repeats=5/20/7, seed=21; ASR=3 independent processes x 10 shuffled paired rounds plus torch. CUDA flags use FP32/exact GELU with no fast math.

Select `--stage gemm|m1|attention|asr` for separate jobs. No adaptive winner selection. The frozen cases and path rules live in boundary_spec.py. M1 baseline builds disable specialization, unlike the old six-way experiment. See TASK.md for the directed-validation rationale.

Offline/read-only commands from repository root:

```bash
python3 hw1-asr/experiments/reuse_boundaries/run_boundaries.py --stage all --dry-run --sm 90
python3 hw1-asr/experiments/reuse_boundaries/run_boundaries.py --stage evidence --history-root /path/to/hw1-asr/experiments --output /tmp/reuse-history-new
python3 -m unittest discover -s hw1-asr/experiments/reuse_boundaries -p test_boundaries.py -v
```

Evidence import verifies frozen hashes and invokes each old run's validators in an isolated process. It produces history_gemm.csv (actual M1 path included), history_attention.csv and history_manifest.json; only eligible warp comparisons are imported. Missing directories are marked missing, never synthesized. Original RAG/API timings cannot be inferred from these files. Historical ASR combination runs remain inventory entries and are not reused as this experiment's system results.

Results default to results/UTC/ with gemm/, m1/, attention/, asr/. Each part generates analysis.md and raw/tabled results; ASR has exact-token checks, all tail samples and shape/operation boundary validation. The two optional report candidate PNGs use matplotlib (install it in the execution environment if absent). Full machine-readable tables include source-level logical load requests and valid output slots. ptxas logs and compiler_resources.csv describe resource use; no occupancy/DRAM claim is inferred without actual counters. Nsight sampling, if available, must be separate from timing runs and save tool version/metric names.

The current host has no nvcc/NVIDIA GPU. Runners and tests can be verified offline; actual correctness, sanitizer and timing outcomes are pending cluster execution. Old system percentages are not reused for new configurations. Kernel numerical tolerances, output guards and H2D stream synchronization come from the existing benchmark implementations.
