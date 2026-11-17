#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
export PATH=/opt/cuda-12.8.0/bin:$PATH
mkdir -p "$HOME/cuda-tmp"
export TMPDIR="$HOME/cuda-tmp"
export NVCC_CCBIN=/usr/bin/g++
export HF_HUB_OFFLINE=1
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
export RAG_ASR_ROOT="$REPO_ROOT/hw1-asr"
source "${RAG_ENV_DIR:-$HOME/mls-rag-env}/bin/activate"
export ASR_CUDA="${ASR_CUDA:-none}"

cd "$REPO_ROOT"
make -C "$RAG_ASR_ROOT" lib

if (( $# )); then
    exec "$@"
fi
exec bash --norc -i
