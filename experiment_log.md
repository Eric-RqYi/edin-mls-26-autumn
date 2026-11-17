### 16-Oct. ASR on GPU: NVIDIA H200 MIG 1g.18gb (sm_90)

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