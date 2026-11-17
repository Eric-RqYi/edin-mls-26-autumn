
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