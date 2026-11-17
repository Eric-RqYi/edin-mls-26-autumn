#!/usr/bin/env python3
"""One independent model process: interleaved timing, exact tokens, diagnostic profiles.

PyTorch is imported only in main(), so validation/profiler tests work without CUDA.
The supplied model, ops and benchmark files are never edited.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import re
import resource
import sys
from collections import defaultdict
from pathlib import Path

from spec import BASELINE

HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]


def normalize(text):
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


def validate_output(tokens, probabilities, transcript, reference, expected):
    if tokens != reference:
        raise ValueError("Generated token IDs differ from the tiled16 reference")
    if len(tokens) != len(probabilities) or not tokens:
        raise ValueError("Missing tokens/probabilities")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Nonfinite/out-of-range selected probability")
    if normalize(transcript) != normalize(expected):
        raise ValueError("Transcription differs from the expected text")


class ShapeProfiler:
    """Temporary instance/module wrappers; restored even if generation raises."""

    def __init__(self, model, ops, torch):
        self.model, self.ops, self.torch = model, ops, torch
        self.phase = "other"
        self.events = []

    def __enter__(self):
        self.original_linear = self.ops.linear
        self.saved = {}
        for name, phase in (("encode_audio", "encoder"), ("next_token_probs", "decoder")):
            self.saved[name] = (name in vars(self.model), vars(self.model).get(name))
            original = getattr(self.model, name)

            def wrapper(*args, _original=original, _phase=phase, **kwargs):
                previous, self.phase = self.phase, _phase
                try:
                    return _original(*args, **kwargs)
                finally:
                    self.phase = previous

            setattr(self.model, name, wrapper)

        def linear(x, weight, bias=None, act=None):
            K, N = int(x.shape[-1]), int(weight.shape[0])
            M = int(x.numel()) // K
            phase = self.phase
            if (phase == "decoder" and M == 1 and N == self.model.c.vocab
                    and K == self.model.c.text_hidden):
                phase = "lm_head"
            key = (phase, M, N, K, int(bias is not None), int(act == "gelu"))
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            y = self.original_linear(x, weight, bias, act)
            end.record()
            self.events.append((key, start, end))
            return y

        self.ops.linear = linear
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.ops.linear = self.original_linear
        for name, (existed, value) in self.saved.items():
            if existed:
                setattr(self.model, name, value)
            else:
                delattr(self.model, name)

    def rows(self):
        self.torch.cuda.synchronize()
        groups = defaultdict(lambda: [0, 0.0])
        for key, start, end in self.events:
            groups[key][0] += 1
            groups[key][1] += start.elapsed_time(end)
        rows = []
        for key, (calls, ms) in groups.items():
            rows.append(dict(zip(("phase", "M", "N", "K", "bias", "act"), key),
                             calls=calls, gpu_event_ms=ms, mean_ms=ms / calls))
        return sorted(rows, key=lambda row: -row["gpu_event_ms"])


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--hw-root", type=Path, default=HW, help="Frozen hw1-asr source snapshot")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing model result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.manifest.read_text())
    variants = meta["variants"]
    if not variants or variants[0]["name"] != BASELINE:
        parser.error("Manifest must start with tiled16")
    os.environ.update(HF_HUB_OFFLINE="1", ASR_DEBUG="0", CUDA_LAUNCH_BLOCKING="0")
    sys.path.insert(0, str(args.hw_root))
    import torch
    from benchmark import EXPECTED, read_wav, use_cache_if_present
    from glm_asr_cuda import ops
    from glm_asr_cuda.model import MODEL_NAME, load_model

    if not torch.cuda.is_available():
        parser.error("No CUDA device visible")
    libraries = {}
    for variant in variants:
        os.environ["ASR_LIB"] = variant["library"]
        ops._lib = None
        libraries[variant["name"]] = ops._library()  # apply original ctypes signatures

    def activate(name):
        ops.set_cuda_groups("none" if name == "torch" else "all")
        if name != "torch":
            ops._lib = libraries[name]  # retain every handle until process exit

    activate(BASELINE)
    use_cache_if_present(MODEL_NAME)
    (model, processor), load_ms = ops.wall_clock_ms(load_model)
    audio = read_wav(args.hw_root / "test_audio.wav")
    inputs = processor.apply_transcription_request(audio)
    model_args = (inputs["input_features"].to("cuda", torch.float32),
                  inputs["input_ids"].to("cuda"), inputs["input_features_mask"].to("cuda"))
    names = [v["name"] for v in variants] + ["torch"]
    rng = random.Random(meta["seed"] + args.block * 100003)
    reference, reference_probs = model.generate(*model_args)
    expected = EXPECTED

    def checked(output):
        tokens, probs = output
        tokens, probs = list(map(int, tokens)), list(map(float, probs))
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        validate_output(tokens, probs, text, reference, expected)
        return tokens, probs, text

    checked((reference, reference_probs))
    print(f"Block {args.block}: loaded model in {load_ms / 1000:.1f}s; reference={reference}", flush=True)
    order = names.copy()
    rng.shuffle(order)
    for name in order:
        activate(name)
        for _ in range(meta["model_warmup"]):
            checked(model.generate(*model_args))
    samples = []
    partial = args.output.with_suffix(".samples.jsonl")
    with partial.open("x") as raw:
        for round_id in range(1, meta["model_runs"] + 1):
            order = names.copy()
            rng.shuffle(order)
            for position, name in enumerate(order):
                activate(name)
                torch.cuda.reset_peak_memory_stats()
                started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
                before = resource.getrusage(resource.RUSAGE_SELF)
                output, ms = ops.wall_clock_ms(model.generate, *model_args)
                after = resource.getrusage(resource.RUSAGE_SELF)
                tokens, probs, text = checked(output)
                if not math.isfinite(ms) or ms <= 0:
                    raise ValueError("Invalid model wall time")
                row = dict(
                    block=args.block, round=round_id, variant=name, order=position, wall_ms=ms,
                    started_utc=started_utc, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    tokens=len(tokens), token_ids=tokens, selected_probs=probs, transcript=text,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cpu_user_ms=1000 * (after.ru_utime - before.ru_utime),
                    cpu_system_ms=1000 * (after.ru_stime - before.ru_stime),
                    voluntary_context_switches=after.ru_nvcsw - before.ru_nvcsw,
                    involuntary_context_switches=after.ru_nivcsw - before.ru_nivcsw,
                )
                samples.append(row)
                raw.write(json.dumps(row, allow_nan=False) + "\n")
                raw.flush()
                print(f"block={args.block} round={round_id} {name}: {ms:.3f} ms ({len(tokens)} tokens)", flush=True)

    # Separate diagnostic generations: none of their timings enter samples above.
    profiles, shapes = [], []
    for name in names:
        activate(name)
        with ShapeProfiler(model, ops, torch) as profiler:
            ops.profile_start()
            try:
                checked(model.generate(*model_args))
            finally:
                operation_table = ops.profile_stop()
        for (operation, backend), (calls, ms) in operation_table.items():
            profiles.append(dict(block=args.block, variant=name, operation=operation,
                                 backend=backend, calls=calls, gpu_event_ms=ms))
        shape_rows = profiler.rows()
        if sum(r["calls"] for r in shape_rows) != sum(
                calls for (op, _), (calls, _) in operation_table.items() if op == "linear"):
            raise ValueError("Shape profile did not cover every linear call")
        shapes.extend(dict(block=args.block, variant=name, **row) for row in shape_rows)
    atomic_json(args.output, dict(
        status="complete", block=args.block, seed=meta["seed"], model_load_ms=load_ms,
        variants=names, reference_tokens=reference, reference_transcript=expected,
        samples=samples, operation_profiles=profiles, shape_profiles=shapes,
    ))
    print(f"Block {args.block} complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
