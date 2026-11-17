#!/usr/bin/env bash
# Submit from hw1-asr. Arguments after this filename go to run_experiment.py.
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=task3-gemm-%j.out
set -eo pipefail
export PATH=/opt/cuda-12.8.0/bin:$PATH
export NVCC_CCBIN="${NVCC_CCBIN:-/usr/bin/g++}"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "${SLURM_SUBMIT_DIR:?Submit from hw1-asr}"
if [[ ! -f experiments/task3_gemm/run_experiment.py ]]; then
    echo "Submit this job from the hw1-asr directory." >&2
    exit 1
fi
python -u experiments/task3_gemm/run_experiment.py "$@"
