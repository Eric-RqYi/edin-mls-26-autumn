# RAG data

Fetch the repository-root `rag-assets.zip` with Git LFS, following
[Step 2 of the RAG guide](../../README.md#step-2-install-the-course-data-once). Extract it into a
folder containing `data/` and `models/`, then run `rag/scripts/install_assets.py`
with `--from-dir` pointing to that folder. See the RAG README for the commands.

This directory contains the corpus, practice/dev WAV files, audio manifest,
public qrels and corpus embeddings for enc_a and enc_b. Every embedding matrix
has its own record-ID mapping. Audio provides the query input; qrels support
evaluation.
