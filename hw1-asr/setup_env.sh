#!/usr/bin/env bash
# One-time setup for hw1-asr. Run it on the head node, because it needs the
# internet and GPU nodes may not have it:
#
#     bash setup_env.sh
#
# It installs Miniconda if you have no conda, creates the conda environment
# "mls-asr" (Python 3.11, PyTorch 2.10 for CUDA 12.8, transformers 5.0), and
# downloads the GLM-ASR-Nano model (about 4.5 GB) into ~/.cache/huggingface.
# In total it needs about 20 GB in your cluster home. Running it again is safe.
set -eo pipefail

ENV_NAME=mls-asr
MINICONDA_DIR="$HOME/miniconda3"

# 1. conda
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -x "$MINICONDA_DIR/bin/conda" ]; then
    eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"
else
    read -rp "No conda found. Install Miniconda into $MINICONDA_DIR? [Y/n] " answer
    if [[ "$answer" =~ ^[nN] ]]; then
        echo "Stopped. Install conda yourself and run this script again."
        exit 1
    fi
    installer="$(mktemp -d)/miniconda.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$installer"
    bash "$installer" -b -p "$MINICONDA_DIR"
    eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"
    conda init bash >/dev/null
fi

# 2. the environment (conda-forge only, so no Anaconda terms of service prompt)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> conda environment $ENV_NAME already exists"
else
    conda create -y -n "$ENV_NAME" --override-channels -c conda-forge python=3.11
fi
conda activate "$ENV_NAME"
# --no-cache-dir: do not keep a second copy of the 3 GB of wheels in ~/.cache/pip
pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir transformers==5.0.0 numpy safetensors huggingface_hub

# 3. the model
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('zai-org/GLM-ASR-Nano-2512'))"

echo
echo ">>> Setup finished. In every new shell, before working on hw1-asr, run:"
echo "    conda activate $ENV_NAME"
