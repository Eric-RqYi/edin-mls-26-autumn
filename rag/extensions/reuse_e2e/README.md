# Task B: paired multi-audio RAG validation

Branch `asr/task-b-rag-e2e`. Experiment code is confined to this directory; the common kernel rollback follows the user's newer instruction over TASK.md. Original baseline explicitly disables M1 and uses tiled16/w4; final enables rb4x4, M1/w4, narrow-N dispatch and attention w16. All flags, source hashes and binary hashes are recorded. Original production RAG/ASR modules and coursework tests are not edited.

From repository root, inspect a plan without GPU/assets/API or output-directory creation:

```bash
python3 rag/extensions/reuse_e2e/run_e2e.py --mode plan --sm 90
```

Optional pre-freeze (requires only the installed manifest/WAVs, not GPU or ELM):

```bash
python3 rag/extensions/reuse_e2e/run_e2e.py --mode freeze --output /absolute/new-cohort-directory
```

Two-call connectivity check, using practice audio only (separate from the main experiment):

```bash
sbatch --gres=gpu:h200_1g.18gb:1 rag/extensions/reuse_e2e/run_cluster.sbatch --mode check
```

Formal run; submit from repository root:

```bash
sbatch --gres=gpu:h200_1g.18gb:1 rag/extensions/reuse_e2e/run_cluster.sbatch --mode run
```

To reuse a pre-frozen cohort add `--sample-manifest /absolute/new-cohort-directory/sample_manifest.json`. The file and every selected audio hash are verified. Default 30 dev items, 10 each from short/medium/long count-balanced duration strata; invalid files are recorded before sampling. Fixed seed 21, three paired rounds, shuffled queries and AB/BA. Fewer valid samples requires explicit `--allow-subset` and is labeled as a limited subset; practice audio never fills dev slots. Exactly two out-of-sample practice files provide two warmups per variant.

The main experiment makes at most **180 formal + 4 warmup client generation calls**. The standalone connectivity check makes two additional calls only when explicitly invoked. OpenAI-compatible SDK retries are disabled, failures retained, no automatic retries/judging/expanded cohort. The generator stores request IDs, status, usage and reported cached-token counts; proxy/server retries or cache behavior not exposed by the service remain unknown. No prompt noise is introduced. This code uses the existing ELM model, system message and temperature=.1, enc_a on CPU, fixed exact retrieval k=5 and original context builder.

The script reuses the validated CUDA compiler/TMPDIR preflight, builds two full independent libraries and runs unchanged coursework tests. Both libraries are explicitly exercised on M=2,N=512,K=2048 under memcheck/racecheck/synccheck **without --small**, before any model/API calls. No historical sanitizer artifact is assumed reusable. No shared transcript cache, cached answer or reconstructed latency is used.

The dedicated worker loads one model and retains both original ctypes-configured library handles. Each response must match request/variant/binary hash. Model weights, enc_a, index and API client startup are recorded separately. For each actual complete request:

- Parent end_to_end_ms spans real audio request through answer return; ASR roundtrip includes file reading, processing, model, IPC and response serialization.
- Worker records audio loading, preprocessing/H2D, synchronized core generate, full token IDs/probabilities, EOS/truncation and peak memory.
- Embedding, retrieval, context construction, live generation and unattributed residual are recorded. Core generation is nested within ASR roundtrip and must not be added twice.
- JSONL collection and vector-to-list conversion happen after the measured request. Server response parsing is part of the actual request.

After formal requests, each stratum's predetermined representative gets separate diagnostic generation for both variants, without ELM calls. It saves actual attention tensor shapes, encoder/decoder phases and original-event boundary checks. Formal rows save feature dimensions and prompt length; duration alone is not treated as actual encoder tensor length. Diagnose padding and generated token counts before explaining shape generalization.

Outputs: metadata.json, sample_manifest.json, request_plan.json, source/ + source_sha256.json, build/ libraries + hashes, worker.log, warmup.jsonl, raw.jsonl, diagnostics.jsonl, per_query.csv, summary.csv/json, output_consistency.csv, retrieval_metrics.csv, analysis.md, and up to two matplotlib figures. All failures and slow successful samples remain. Summary uses queries as clusters: reduce three pairs per query first, then aggregate by stratum/overall, with fixed-seed query bootstrap. Complete-pair coverage is reported; incomplete requests are never silently replaced. Input inconsistency or failed requests produce status incomplete_or_mismatch and exit 2, rather than a quality-preservation success claim.

Rebuild derived analysis without network requests:

```bash
python3 rag/extensions/reuse_e2e/summarize_e2e.py /absolute/result-directory
python3 -m unittest discover -s rag/extensions/reuse_e2e -p test_e2e.py -v
```

Embedding and score comparisons use atol=1e-6, rtol=1e-5; token IDs, transcript, ranked IDs, context and prompt require equality across both variants and all repeats. Recall@5/MRR@5/NDCG@5 use public qrels. No manual reference transcription or answer judge is used, so no WER/CER or verified answer-quality score is claimed. Prompt equality does not imply deterministic ELM answers. A confidence interval crossing zero is reported as unresolved full-system savings under API variability.

Local execution prerequisites currently absent: nvcc/NVIDIA GPU, installed course manifest/audio/corpus/embeddings, enc_a weights, an ASR cache/interpreter and usable ELM configuration. No performance or API result is fabricated. On the cluster follow rag/README.md for asset installation and RAG environment; set CONDA_BASE/RAG_VENV if their locations differ, keep secrets in rag/.env, and provision matplotlib for figures. Default job limit is 8 hours and can be overridden by sbatch. A timeout preserves partial raw records; no automatic resume makes extra requests.
