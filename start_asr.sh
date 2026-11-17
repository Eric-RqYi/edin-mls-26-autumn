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
export ASR_CUDA="${ASR_CUDA:-all}"

cd "$REPO_ROOT/hw1-asr"
# make lib

if (( $# )); then
    exec "$@"
fi
exec bash --norc -i
