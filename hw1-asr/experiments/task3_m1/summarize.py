#!/usr/bin/env python3
"""Validate experiment coverage and summarize real cluster measurements (stdlib only)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

CASE_NAMES = (
    "m1_scalar", "m1_short", "m1_bias_k31", "m1_gelu_k32", "m1_bias_gelu_tail",
    "m1_k129", "m1_long_odd", "m1_large_k", "lm_head", "fallback_m2", "fallback_m3",
)


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())


def summarize_micro(path: Path, variants: list[str], repeats: int) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    seen = set()
    shapes = {}
    for row in rows:
        case, variant, repeat = row["case"], row["variant"], int(row["repeat"])
        key = (case, variant, repeat)
        if case not in CASE_NAMES or variant not in variants or not 1 <= repeat <= repeats:
            raise ValueError(f"Unexpected microbenchmark sample: {key}")
        if key in seen:
            raise ValueError(f"Duplicate microbenchmark sample: {key}")
        seen.add(key)
        shape = tuple(int(row[k]) for k in ("M", "N", "K", "bias", "act"))
        if case in shapes and shapes[case] != shape:
            raise ValueError(f"Inconsistent shape for {case}")
        shapes[case] = shape
        for field in ("ms", "weight_gbs", "gflops", "max_abs_error"):
            number = float(row[field])
            if not math.isfinite(number) or number < 0 or (field == "ms" and number == 0):
                raise ValueError(f"Invalid {field} for {key}")
        if int(row["wrong"]) != 0:
            raise ValueError(f"Incorrect kernel for {key}")
        grouped[case, variant].append(row)
    expected = {(c, v, r) for c in CASE_NAMES for v in variants for r in range(1, repeats + 1)}
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"Incomplete microbenchmark CSV; missing {missing[:5]}")

    summary = []
    for case in CASE_NAMES:
        baseline = {int(r["repeat"]): float(r["ms"]) for r in grouped[case, "tiled"]}
        for variant in variants:
            samples = grouped[case, variant]
            times = [float(r["ms"]) for r in samples]
            summary.append(dict(
                case=case, variant=variant,
                **dict(zip(("M", "N", "K", "bias", "act"), shapes[case])),
                samples=len(times), median_ms=statistics.median(times),
                mean_ms=statistics.mean(times),
                stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
                min_ms=min(times), max_ms=max(times),
                paired_speedup_median=statistics.median(
                    baseline[int(r["repeat"])] / float(r["ms"]) for r in samples),
                weight_gbs_median=statistics.median(float(r["weight_gbs"]) for r in samples),
                max_abs_error=max(float(r["max_abs_error"]) for r in samples),
            ))
    return summary


def parse_model_log(path: Path, expected_runs: int) -> dict:
    text = path.read_text()
    runs = re.findall(r"^\s*run (\d+):\s*([\d.]+) ms\s*\((\d+) tokens\)", text, re.M)
    if [int(r[0]) for r in runs] != list(range(1, expected_runs + 1)):
        raise ValueError(f"Missing or duplicate model runs in {path}")
    times = [float(r[1]) for r in runs]
    counts = {int(r[2]) for r in runs}
    if any(not math.isfinite(t) or t <= 0 for t in times) or len(counts) != 1:
        raise ValueError(f"Invalid timing or changing token count in {path}")

    def field(name: str) -> str:
        matches = re.findall(rf"^{re.escape(name)}:\s*(.+)$", text, re.M)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {name} field in {path}")
        return matches[0].strip()

    transcript, expected = field("Transcription"), field("Expected")
    if field("Status") != "PASS" or normalize(transcript) != normalize(expected):
        raise ValueError(f"Model transcription does not match expected text in {path}")
    linear = re.findall(r"^\s*linear\s+(?:cuda|torch)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%", text, re.M)
    if len(linear) != 1:
        raise ValueError(f"Missing linear profile in {path}")
    return dict(
        samples=len(times), median_ms=statistics.median(times), mean_ms=statistics.mean(times),
        stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=min(times), max_ms=max(times), tokens=counts.pop(), transcript=transcript,
        peak_gpu_gib=float(field("Peak GPU memory").split()[0]),
        linear_calls=int(linear[0][0]), linear_profile_ms=float(linear[0][1]),
        linear_profile_share_pct=float(linear[0][2]),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output: Path) -> Path:
    # A failed re-summary must not leave an older success report looking current.
    for name in ("summary.md", "micro_summary.csv", "model_summary.csv"):
        (output / name).unlink(missing_ok=True)
    metadata = json.loads((output / "metadata.json").read_text())
    variants = [v["name"] for v in metadata["variants"]]
    if not variants or variants[0] != "tiled" or len(set(variants)) != len(variants):
        raise ValueError("Metadata must contain a unique tiled baseline first")
    micro = summarize_micro(output / "micro.csv", variants, metadata["repeats"])
    model = []
    if not metadata["skip_model"]:
        for name in ["torch", *variants]:
            model.append(dict(variant=name, **parse_model_log(
                output / "model" / f"{name}.log", metadata["model_runs"])))
        baseline = next(row for row in model if row["variant"] == "tiled")
        for row in model:
            if (normalize(row["transcript"]) != normalize(baseline["transcript"])
                    or row["tokens"] != baseline["tokens"]):
                raise ValueError(f"Transcript/token-count mismatch for {row['variant']}")
            row["speedup_vs_tiled"] = baseline["median_ms"] / row["median_ms"]

    # Publish summaries only after all enabled checks have passed.
    write_csv(output / "micro_summary.csv", micro)
    lines = [
        "# Task 3 M=1 experiment", "",
        "All configured correctness/coverage checks passed. Raw logs remain authoritative.", "",
        "## lm_head: 1 x 59264 x 2048", "",
        "| Variant | Median ms | Stddev ms | Paired speedup vs tiled | Weight GB/s proxy | Max abs error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in micro:
        if row["case"] == "lm_head":
            lines.append(f"| {row['variant']} | {row['median_ms']:.6f} | {row['stdev_ms']:.6f} | "
                         f"{row['paired_speedup_median']:.3f}x | {row['weight_gbs_median']:.2f} | "
                         f"{row['max_abs_error']:.3e} |")
    lines += ["", "Micro samples are CUDA-event batch means with shuffled variant order per round.",
              "Weight GB/s = N*K*4/time; it is a logical proxy, not measured DRAM bandwidth.",
              "micro_summary.csv contains all shapes, including M>1 fallback controls.", ""]
    if model:
        write_csv(output / "model_summary.csv", model)
        lines += ["## Full ASR on the supplied audio", "",
                  "| Variant | Median ms | Stddev ms | Speedup vs tiled | Tokens | Linear profile ms | Peak GiB |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in model:
            lines.append(f"| {row['variant']} | {row['median_ms']:.1f} | {row['stdev_ms']:.2f} | "
                         f"{row['speedup_vs_tiled']:.3f}x | {row['tokens']} | "
                         f"{row['linear_profile_ms']:.1f} | {row['peak_gpu_gib']:.2f} |")
        lines += ["", "Timings exclude model loading and audio preprocessing; profiling is a separate run.",
                  "ASR variants run in separate processes; model speedups are ratios of medians, not paired samples.",
                  "The final timed transcription is checked against expected text and the tiled baseline.",
                  "Token counts are checked for every timed run; this does not establish identical token IDs.",
                  "Only one supplied audio is evaluated; this is not a WER/CER evaluation."]
    else:
        lines += ["Full-model evaluation was explicitly skipped. No ASR latency/transcription claim is established."]
    path = output / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="An existing experiment result directory")
    args = parser.parse_args()
    try:
        print(summarize(args.output.resolve()))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Summary failed: {error}\n")


if __name__ == "__main__":
    main()
