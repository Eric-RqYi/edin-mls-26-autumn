#!/bin/bash
# Batch version of "make test && ./run.sh --profile". From hw1-asr, on the head node:
#     sbatch run_asr.sbatch
#     squeue -u "$USER"          # wait until the job is gone
#     cat asr-JOB_ID.out
#SBATCH -p Teaching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=asr-%j.out

export PATH=/opt/cuda-12.8.0/bin:$PATH
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
cd "$SLURM_SUBMIT_DIR"

hostname
nvidia-smi -L
make test
./run.sh --profile
