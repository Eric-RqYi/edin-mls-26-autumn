### 18-Nov-2026: Full-ASR factorial results

Run: `hw1-asr/experiments/task4_attention/results/20261118T082810Z-1093639`.
Hardware: H200 MIG 1g.18gb, sm_90, 16 SMs. All comparisons below use this run's
paired baseline (rb4x4 + M=1/w4, attention w4/standard).

- Combined ASR median: 3371.26 -> 3074.82 ms, saving 296.44 ms (8.79% latency
  reduction), median block-paired speedup 1.096366x. Three block ratios range
  from 1.096333x to 1.096482x. Mean is 3075.03 ms; no CUDA sample exceeds its
  block median by 5%, and every raw sample remains included.
- Hybrid alone saves 35.17 ms (1.04%); attention w16/direct alone saves
  260.35 ms (7.72%). Combined profile savings are 35.49 ms linear + 260.30 ms
  attention, matching wall-time savings within about 0.65 ms. Median paired
  interaction is -0.456 ms: approximately additive, without evidence for a
  meaningful extra synergy.
- Direct stage profiling confirms encoder attention falls from 955.53 to
  704.23 ms (32 calls), decoder from 33.27 to 24.27 ms (364 calls). Encoder
  accounts for about 96.63% of baseline attention, matching the earlier budget.
- Read-only audit: 33 frozen source hashes, 392 attention micro samples,
  924 GEMM samples, 150 timed model samples, 16 coursework task-test runs and
  six sanitizer logs checked. All 150 token sequences agree (13 tokens each).
  All 30 linear/attention shape-to-operation boundary checks match exactly.
- Hybrid performance depends on context: M=65/80/96 is slower in hot-cache
  KV microbenchmarks, while actual model M=65..71 still saves about 3.35 ms.
  Do not narrow the dispatch rule solely from one microbenchmark context.
- Coverage finding: GEMM sanitizer --small only selects six cases with
  M*N*K<=1e5, none on the N=512/K=2048/M>1 hybrid route. Numerical/model checks
  pass; a dedicated KV-path sanitizer set remains a next-step validation item.
- Next experimental baseline: combined. After dedicated sanitizer coverage,
  prioritize linear (75.53% of combined operation profile), particularly
  decoder gate/up+down (~1145.74 ms, 37.26% of ASR). A small K-tile 16/32 x RN
  4/8 ablation is proposed; no new tuning kernel or default was applied here.

Analysis, reproducible audit and derived tables: `analysis/` inside this run.
Original measurements, reports and frozen sources were preserved.

### 18-Nov-2026: Attention sweep results and repeat comparison

Retained run: `hw1-asr/experiments/task4_attention/results/20261118T001456Z-748564`.
Compared repeat: `20261118T001457Z-748568`, removed at the user's request after
confirming the same experiment conclusions. Both ran on H200 MIG 1g.18gb
(sm_90, 16 SMs), on different physical GPUs and MIG UUIDs.

- Configurations, compiler flags and all 33 frozen source hashes match. Each
  run has 980 attention and 280 GEMM samples with complete coverage, zero wrong
  outputs, 40 coursework test passes and six clean small-case sanitizer checks.
- Across 140 attention case/variant medians, the absolute run-to-run difference
  has median 0.317% and maximum 1.590%; for 40 GEMM control groups, median 0.037%
  and maximum 0.776%. All 14 attention case winners agree. Small timing/ranking
  differences remain; this is not a formal statistical equivalence claim.
- Retained encoder result: w4_standard 29.861115 ms -> w16_direct 22.005772 ms,
  paired speedup 1.356967x, latency reduced 26.31%. Warp count supplies most of
  this: w16_standard is 22.602106 ms; direct saves a further 2.64% at 16 warps.
- Decoder prefers w16_standard (0.049563 ms) to w16_direct (0.056541 ms), so the
  encoder winner is not assumed to be best for every attention shape. Standard
  uses 40 registers/thread, direct 32; all attention configurations have no spills.
- These are microbenchmarks only (`skip_model=true`); full-ASR latency, token
  consistency and stage profiles have not been established for the new configs.
  The 32-encoder-call budget is about 251 ms saved, not a measured system gain.
- Next: run the existing factorial stage with `--selected-warps 16
  --selected-online direct --sanitizer`, retaining three model blocks and ten
  paired rounds. Compare hybrid/attention alone and together, exact tokens,
  shape-event boundaries and all slow samples before changing defaults.
- Saved both-run summaries, metadata/audits, comparison CSV and the removed-file
  SHA256 inventory before deleting the later full directory (195 files,
  79,558,328 bytes). The retained raw measurements and frozen sources were
  preserved. The analysis audit also runs after deletion using saved comparison
  records for the removed group.

Full report, comparison, audit script and cleanup record:
`analysis/` inside the retained run.

### 17-Nov-2026: Attention sweep and GEMM-dispatch factorial (prepared offline)

- Added independent `ASR_ATTN_WARPS` and `ASR_ATTN_ONLINE_MODE` controls in
  Task 4. The direct recurrence avoids duplicate probability exponentials;
  all-masked query rows explicitly output zero, with shared-memory padding
  and uniform tile-loop barriers retained. Default stays w4/standard.
- Added opt-in `ASR_TASK3_HYBRID_KV`: M>1, N=512, K=2048 uses tiled32 after
  the dedicated M=1 branch; other shapes use the configured GEMM path.
- New `experiments/task4_attention/run_cluster.sbatch` runs a ten-configuration
  warp/softmax sweep, then a four-group hybrid x selected-attention confirmation.
  All experiment variants fix rb4x4 and M=1/w4. Sweep defaults to microbenchmarks;
  factorial defaults to three paired-model processes with ten rounds each.
- Added 14 independent-reference attention cases, including GQA/cache, tails,
  late maxima, peaked scores and fully masked prefixes; factorial also runs
  33 GEMM cases with held-out M and neighboring N/K controls. Unchanged coursework
  tests run for every library; small-case sanitizer checks are optional.
- Linear and attention shape profiles now reuse the original ops events, with
  attention stages/shapes and schema v2 saved in model JSON. The new summary
  rejects mismatched event sums/call counts and checks every timed token sequence.
  All slow samples remain, with >5% and >2x diagnostic flags and paired interaction
  effects for the four-group comparison.
- Reused the validated host-compiler/TMPDIR preflights, frozen source hashes,
  separate builds, shuffled measurements, telemetry and failure-state handling.
- Offline validation: 17 attention-tool tests, 23 GEMM-tool tests and 17 M=1-tool
  tests passed; Bash syntax and dry-run checks passed. Actual CUDA compilation,
  numerical correctness, sanitizer output and speedups remain pending cluster
  execution. No experimental configuration was made the production default.

### 17-Nov-2026: Completed M>1 GEMM run and analysis

Run: `hw1-asr/experiments/task3_gemm/results/20261117T192802Z-615737`.
Hardware: metadata, GPU/test/sanitizer logs, build targets and telemetry identify
H200 MIG 1g.18gb (sm_90, 16 SMs). The user confirmed that analysis should use
these saved logs; this run is analyzed and reported as H200 MIG.

- Complete coverage: 924 micro samples, 210 model samples, six variants passing
  all coursework tasks, small-case memcheck/racecheck/synccheck clean, 28 frozen
  source hashes matching. Exact token IDs agree on every timed generation.
- rb4x4: ASR median 3381.82 ms versus tiled16 5942.23 ms; latency reduced 43.09%,
  median block-paired speedup 1.75714x. All three block ratios agree closely.
- Linear profile falls from 4923.24 to 2363.45 ms, saving 2559.79 ms, versus
  2560.42 ms measured model-median saving. Attention remains about 996 ms.
- Encoder matrices gain about 2.7x. Decoder narrow-N K/V prefers tiled32;
  diagnostic hybrid dispatch estimates only about 37 ms (1.1%) extra headroom
  among tested candidates, requiring a direct model comparison to validate.
- After rb4x4, linear/attention shares are about 69.85%/29.44%. Next main target:
  encoder attention warp-count and online-softmax experiments; hybrid GEMM
  dispatch is a small secondary ablation.
- Preserve tails: rb2x4 has a 6480.18 ms sample (about 1.706x its block median)
  despite the old >2x slow flag being zero. Diagnostic GPU-event boundaries also
  differ, so shape-profile estimates are not pure kernel measurements.

Full analysis and next plan: `analysis/analysis_and_next_plan.md` inside this run.
Audit/derived metrics and supporting tables are in the same analysis directory.
No kernel defaults or original measurements were changed during analysis.

### 17-Nov-2026: GEMM build failure and compiler-environment fix

- Run `20261117T171901Z-590775` stopped in the first tiled16 build, while compiling
  Task 1/2: `nvcc fatal: Failed to preprocess host compiler properties`. No new
  Task 3 compilation, correctness result or timing was reached.
- The batch entry omitted the TMPDIR configuration present in `start_asr.sh`.
  The quickstart records that `TMPDIR=$HOME/cuda-tmp` plus `/usr/bin/g++` had
  previously compiled successfully; it did not isolate the two settings. The
  failed run did not record TMPDIR, so its exact underlying cause is unconfirmed.
- Aligned both sbatch and direct Python execution with that setup, using a
  private nvcc scratch directory per run. Record inherited/effective TMPDIR and
  compiler, and restore the caller's Python environment on exit.
- Added host compiler version/preprocessing and minimal CUDA compile preflights
  before variant builds, with verbose nvcc output and retained intermediates.
- Local validation: 22 tool tests passed, including actual local C++ preprocessing,
  environment restoration and early-stop regression checks; sbatch syntax and
  dry-run passed. Actual nvcc/GPU validation still requires a new cluster run.
- Original failed-run logs and frozen sources were preserved.

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
