#!/usr/bin/env bash
# Build your kernels for this GPU, then transcribe test_audio.wav with them.
#
#   ./run.sh                       every task on your CUDA kernels
#   ./run.sh --cuda none           PyTorch reference only (checks your setup)
#   ./run.sh --cuda rope,reduce    only these groups on CUDA (task1..task4 also work)
#   ./run.sh --profile             also show how long each operation takes
#   ./run.sh --tokens              also show each generated token and its probability
set -e
cd "$(dirname "$0")"

# The PyTorch-only run does not need the kernels, so it works even if they do not compile.
if [[ " $* " != *" --cuda none "* && " $* " != *" --cuda=none "* ]]; then
    make -s lib
fi
python benchmark.py "$@"
