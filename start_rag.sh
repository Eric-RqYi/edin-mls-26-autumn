#!/usr/bin/env bash
# Run with bash (do not source). See gpu_cluster_quickstart.md.
set -eo pipefail

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "请用 bash ${BASH_SOURCE[0]} 启动，不要 source。" >&2
    return 1
fi
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'bash start_rag.sh [command [args...]] — 无参数时进入 RAG shell；工作目录为仓库根目录。'
    exit 0
fi
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Leave any inherited venv before selecting the ASR conda environment.
if declare -F deactivate >/dev/null; then deactivate; fi
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
else
    echo '未找到 conda；先在 head node 运行 bash hw1-asr/setup_env.sh，或设置 CONDA_BASE。' >&2
    exit 1
fi
conda activate "${ASR_ENV_NAME:-mls-asr}"
ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"

export CUDA_HOME="${CUDA_HOME:-/opt/cuda-12.8.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export ASR_CUDA="${ASR_CUDA:-none}"
# Keep CUDA_VISIBLE_DEVICES as supplied by Slurm, including MIG device IDs.
command -v nvcc >/dev/null || { echo "未找到 nvcc；检查 CUDA_HOME=$CUDA_HOME" >&2; exit 1; }
hostname
nvidia-smi -L
nvcc --version
"$ASR_PYTHON" - <<'PYTHON'
import torch
from huggingface_hub import snapshot_download
if not torch.cuda.is_available():
    raise SystemExit('没有可用的 CUDA GPU；请先通过 srun 进入 GPU 节点。')
print(f'ASR GPU: {torch.cuda.get_device_name()} / sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}')
try:
    snapshot_download('zai-org/GLM-ASR-Nano-2512', local_files_only=True)
except Exception as exc:
    raise SystemExit('本地 ASR 模型缓存不可用；先在 head node 运行 hw1-asr/setup_env.sh。') from exc
PYTHON

# Capture the ASR interpreter BEFORE activating the separate RAG venv.
export RAG_ASR_PYTHON="$ASR_PYTHON"
export RAG_ASR_ROOT="${RAG_ASR_ROOT:-$REPO_ROOT/hw1-asr}"
RAG_ENV_DIR="${RAG_ENV_DIR:-$HOME/mls-rag-env}"
if [[ ! -f "$RAG_ENV_DIR/bin/activate" ]]; then
    echo "未找到 RAG 环境：$RAG_ENV_DIR；请先完成 rag/README.md 的 Step 1。" >&2
    exit 1
fi
source "$RAG_ENV_DIR/bin/activate"
cd "$REPO_ROOT"
make -C "$RAG_ASR_ROOT" lib
# Reads rag/.env via python-dotenv; never execute it as shell code or print keys.
# This check makes no ELM requests.
python rag/scripts/check_setup.py
printf '\nRAG 已就绪：ASR_CUDA=%s\nASR Python=%s\nRAG Python=%s\n' \
    "$ASR_CUDA" "$RAG_ASR_PYTHON" "$(command -v python)"
if (( $# )); then
    exec "$@"
fi
printf '可运行：python rag/scripts/run_audio_query.py --device cpu\n退出此 shell 后返回原 GPU shell；再 exit 释放 Slurm 资源。\n'
exec bash --norc -i
