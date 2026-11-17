#!/usr/bin/env bash
# Submit from hw1-asr: sbatch experiments/task3_m1/run_cluster.sbatch
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=task3-m1-%j.out
set -eo pipefail

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr

# sbatch copies this file into a spool directory; resolve from submission cwd.
cd "${SLURM_SUBMIT_DIR:?Submit this job from hw1-asr}"
if [[ ! -f experiments/task3_m1/run_experiment.py ]]; then
    echo "Submit from the hw1-asr directory." >&2
    exit 1
fi
python -u experiments/task3_m1/run_experiment.py "$@"
