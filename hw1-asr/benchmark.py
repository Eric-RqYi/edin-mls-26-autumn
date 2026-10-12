#!/usr/bin/env python3
"""Transcribe test_audio.wav with GLM-ASR-Nano running on your kernels.

Usually started through ./run.sh, which builds the kernels first:

    ./run.sh                       every task on your CUDA kernels
    ./run.sh --cuda none           PyTorch reference only (checks your setup)
    ./run.sh --cuda rope,reduce    only these groups on CUDA (task1..task4 also work)
    ./run.sh --profile             also show how long each operation takes
    ./run.sh --tokens              also show each generated token and its probability
"""

import argparse
import os
import re
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED = "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS"


def read_wav(path):
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2 or f.getframerate() != 16000:
            sys.exit(f"{path}: expected 16-bit audio at 16 kHz")
        audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)
        audio = audio.reshape(-1, f.getnchannels()).mean(axis=1)
    return audio / 32768.0


def word_accuracy(text, expected):
    """Share of the expected words that appear in the transcription."""
    def words(s):
        return set(re.sub(r"[^\w\s]", "", s.upper()).split())
    exp = words(expected)
    return len(words(text) & exp) / len(exp)


def use_cache_if_present(model_name):
    """Compute nodes may have no internet. If the model is already downloaded, stay offline."""
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(model_name, local_files_only=True)
        os.environ["HF_HUB_OFFLINE"] = "1"
    except Exception:
        print("The model is not downloaded yet; trying to download it (needs internet).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--cuda", default=os.environ.get("ASR_CUDA", "all"),
                        help="kernel groups on CUDA: all, none, or e.g. rope,reduce (default: all)")
    parser.add_argument("--audio", default=str(HERE / "test_audio.wav"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--profile", action="store_true", help="time each operation")
    parser.add_argument("--tokens", action="store_true", help="print tokens with probabilities")
    args = parser.parse_args()

    import torch
    from glm_asr_cuda import ops
    from glm_asr_cuda.model import MODEL_NAME, load_model

    if not torch.cuda.is_available():
        sys.exit("No GPU visible. Run this on a GPU node (see README.md, step 2).")
    ops.set_cuda_groups(args.cuda)
    major, minor = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} (sm_{major}{minor})")
    print(f"Kernels: {ops.describe()}")
    if ops.USE_CUDA:
        print(f"Library: {ops.library_path()}")

    use_cache_if_present(MODEL_NAME)
    print("Loading the model (about 9 GB of float32 weights)...")
    (model, processor), load_ms = ops.wall_clock_ms(load_model)
    print(f"Loaded in {load_ms / 1000:.1f} s")

    audio = read_wav(args.audio)
    inputs = processor.apply_transcription_request(audio)
    features = inputs["input_features"].to("cuda", torch.float32)
    input_ids = inputs["input_ids"].to("cuda")
    features_mask = inputs["input_features_mask"].to("cuda")
    print(f"Audio: {len(audio) / 16000:.2f} s, mel features {tuple(features.shape)}, "
          f"prompt {input_ids.shape[1]} tokens")

    for _ in range(args.warmup):
        model.generate(features, input_ids, features_mask)
    times = []
    for i in range(args.runs):
        (tokens, probs), ms = ops.wall_clock_ms(model.generate, features, input_ids, features_mask)
        times.append(ms)
        print(f"  run {i + 1}: {ms:8.1f} ms  ({len(tokens)} tokens)")

    text = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    acc = word_accuracy(text, EXPECTED)
    passed = acc > 0.8

    if args.tokens:
        print("\nGenerated tokens and the probability the model gave each one:")
        for t, p in zip(tokens, probs):
            print(f"  {processor.tokenizer.decode([t])!r:>14}  {p:6.3f}")

    if args.profile:
        ops.profile_start()
        model.generate(features, input_ids, features_mask)
        table = ops.profile_stop()
        total = sum(ms for _, ms in table.values())
        print(f"\nGPU time per operation, one transcription ({total:.1f} ms in kernels):")
        print(f"  {'operation':<12} {'runs on':<8} {'calls':>7} {'total ms':>10} {'share':>7}")
        for (op, backend), (calls, ms) in sorted(table.items(), key=lambda kv: -kv[1][1]):
            print(f"  {op:<12} {backend:<8} {calls:>7} {ms:>10.1f} {100 * ms / total:>6.1f}%")

    print(f"\nTime: {np.mean(times):.1f} ms per transcription (+/- {np.std(times):.1f} ms), "
          f"{np.mean(times) / max(len(tokens), 1):.1f} ms per token")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"Transcription: {text}")
    print(f"Expected:      {EXPECTED.capitalize()}")
    print(f"Accuracy: {acc * 100:.1f}%")
    print("Status: PASS" if passed else "Status: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
