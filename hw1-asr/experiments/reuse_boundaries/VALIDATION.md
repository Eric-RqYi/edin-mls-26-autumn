# Local delivery validation — 18 November 2026

- 8 new offline tests passed, including explicit M1 path attribution, complete raw coverage rejection, sanitizer early-stop and environment restoration, frozen sources, and read-only dry-run.
- Existing GEMM (23), M1 (17), and attention-tool (17) regression tests passed in separate processes.
- Bash syntax and Python 3.11 grammar checks passed.
- Historical data were available in the original checkout: 132 six-configuration GEMM summary rows and 70 eligible attention rows were revalidated with their own frozen validators. Derived tables are in results/history-20261118-verified/ on this machine; they are intentionally ignored by Git. The old M1 run lacks source_sha256.json and is marked not imported, not synthesized.
- New CUDA compilation, sanitizer and model results remain pending: no nvcc/NVIDIA GPU or ASR runtime is available locally. No new GPU speedup is claimed.

Run the new cluster entry in README.md. Default history-root refers to the active checkout; when using separate worktrees, pass --history-root pointing to the checkout containing old results. Source datasets, measurements and the original dirty checkout were preserved.
