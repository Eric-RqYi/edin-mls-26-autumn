
### 17-Nov-2026: Next-round M>1 GEMM experiments (prepared offline)

- Added configurable TILE=8/16/32 and register-blocked 2x2/2x4/4x4 output-per-thread
  kernels in Task 3. Default remains tiled16 + the previously tested M=1/w4 path.
- New entry point, from `hw1-asr`:
  `sbatch experiments/task3_gemm/run_cluster.sbatch --sanitizer`.
- Six variants fix M=1 to four warps. The experiment freezes sources before
  building, runs unchanged coursework tests and 22 additional GEMM cases, and
  optionally checks small cases with memcheck/racecheck/synccheck.
- Three independent model processes load weights once each and interleave
  variants within ten timing rounds. Exact token IDs and final text are checked
  on every generation; CPU usage, context switches and UTC timestamps are saved.
- Separate diagnostic generations aggregate linear cost by phase and actual
  shape; best-effort physical GPU telemetry supports investigation of stalls.
  No slow samples are removed and no candidate is automatically deployed.
- Offline validation: 18 new tool tests and 17 previous M=1 tool tests passed;
  coverage includes profile-wrapper restoration, malformed-data rejection,
  dry-run commands and sbatch syntax. CUDA compilation, GPU
  correctness and performance remain pending cluster execution.

### 17-Nov-2026: Task 3 M=1 cluster results

Run: `hw1-asr/experiments/task3_m1/results/20261117T154855Z-483031`.
GPU: H200 MIG 1g.18gb, 16 SMs, CUDA 12.8. All five variants passed the
four coursework tests; 385 microbenchmark records have complete coverage and
zero wrong values. Small-case memcheck and synccheck reported zero errors.

- lm_head: tiled median 4.059883 ms; m1_w4 median 0.886355 ms, about 4.58x faster.
- Full ASR: tiled median 5983.15 ms; m1_w4 median 5940.05 ms, 43.10 ms or 0.720%
  less latency. All configurations produced 13 tokens and the expected final transcript.
- Thirteen lm_head calls predict 41.26 ms saving; the measured linear profile
  decreased by 41.90 ms. The small system gain is consistent with the original
  lm_head share of about 0.882%.
- m1_w2 differs from m1_w4 by only about 0.0081% in the microbenchmark; keep the
  default at four warps rather than declaring a winner from this run.
- Preserve the m1_w8 outlier: run 5 took 16283.0 ms, versus a 5940.75 ms median.
  Its cause is unresolved; do not omit it when reporting mean/variability.
- Next priority: independently repeat the baseline comparison, profile linear
  by phase/shape, then optimize M>1 GEMM. Encoder attention is the second main
  target. Current linear and attention profile shares are 82.8% and 16.8%.

Detailed analysis and next experiment plan: `analysis/analysis.md` inside this run.
Independent audit/derived metrics: `analysis/analyze_results.py` and
`analysis/derived_metrics.json`. Original measurements were left unchanged.

### 17-Nov-2026: Task 3 M=1 specialization (prepared offline)

Branch: `ASR-improvement`. Research question: can a warp-cooperative M=1
matrix-vector kernel reduce lm_head and full ASR latency while preserving
numerical correctness and transcription?

- Added M=1 dispatch in `hw1-asr/csrc/task3_gemm.cu`; M>1 retains the existing
  TILE=16 kernel. Bias, GELU and the caller's CUDA stream remain supported.
- Baseline switch: `ASR_TASK3_M1_ENABLED=0`; optimized switch: `=1`.
  Default `ASR_TASK3_M1_WARPS=4` is a starting configuration, not a measured winner.
- Added `hw1-asr/experiments/task3_m1/`: isolated builds for tiled and 1/2/4/8
  warp variants, unchanged coursework tests, independent CPU-reference checks,
  optional memcheck/synccheck, shuffled CUDA-event measurements, full-model
  comparisons, source snapshots and CSV/Markdown summaries.
- Cluster entry point, from `hw1-asr`:
  `sbatch experiments/task3_m1/run_cluster.sbatch --sanitizer`.
- Local validation: Python tool tests, command dry-run and sbatch syntax checks.
  No local nvcc or NVIDIA GPU is available. CUDA compilation, GPU correctness,
  sanitizer results and speedups are **pending cluster execution**; no performance
  improvement is claimed at this stage.

### 16-Oct ASR on GPU: NVIDIA H200 MIG 1g.18gb (sm_90)

Baseline: finished blanks in four tasks
```
(mls-asr) [saxa]s2955206: ./run.sh
GPU: NVIDIA H200 MIG 1g.18gb (sm_90)
Kernels: rope=cuda  reduce=cuda  gemm=cuda  attention=cuda  provided=cuda
Library: /home/s2955206/edin-mls-26-autumn/hw1-asr/build/sm_90/libasr_kernels.so
Loading the model (about 9 GB of float32 weights)...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loaded in 95.6 s
Audio: 3.50 s, mel features (1, 128, 3000), prompt 59 tokens
  run 1:   5977.4 ms  (13 tokens)
  run 2:   5977.5 ms  (13 tokens)
  run 3:   5977.2 ms  (13 tokens)

Time: 5977.3 ms per transcription (+/- 0.1 ms), 459.8 ms per token
Peak GPU memory: 8.56 GiB
Transcription: Concord returned to its place amidst the tents.
Expected:      Concord returned to its place amidst the tents
Accuracy: 100.0%
Status: PASS
```
