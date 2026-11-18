#!/usr/bin/env bash
# Submit from hw1-asr. Remaining arguments go to run_attention.py.
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=task4-attention-%j.out
set -eo pipefail
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
export PATH=/opt/cuda-12.8.0/bin:$PATH
export NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++}"
# Preserve the compiler setup validated on the cluster after the earlier failure.
export ASR_CUDA_TMPDIR="${ASR_CUDA_TMPDIR:-$HOME/cuda-tmp}"
mkdir -p "$ASR_CUDA_TMPDIR"
export TMPDIR="$ASR_CUDA_TMPDIR"
cd "${SLURM_SUBMIT_DIR:?Submit from hw1-asr}"
if [[ ! -f experiments/task4_attention/run_attention.py ]]; then
    echo "Submit this job from the hw1-asr directory." >&2
    exit 1
fi
python -u experiments/task4_attention/run_attention.py "$@"
