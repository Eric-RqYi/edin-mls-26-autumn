#!/usr/bin/env bash
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=reuse-boundaries-%j.out
set -eo pipefail
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
export PATH=/opt/cuda-12.8.0/bin:$PATH
export NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++}"
export ASR_CUDA_TMPDIR="${ASR_CUDA_TMPDIR:-$HOME/cuda-tmp}"
mkdir -p "$ASR_CUDA_TMPDIR"
export TMPDIR="$ASR_CUDA_TMPDIR"
cd "${SLURM_SUBMIT_DIR:?Submit from hw1-asr}"
test -f experiments/reuse_boundaries/run_boundaries.py
python -u experiments/reuse_boundaries/run_boundaries.py "$@"
