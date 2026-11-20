#!/usr/bin/env python3
"""Strict coverage/correctness checks and descriptive experiment summaries."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from model_runner import normalize, validate_output
from spec import BASELINE, CASE_FIELDS


def number(value, positive=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid measurement: {value}")
    return value


def stats(values):
    values = sorted(values)
    if not values:
        raise ValueError("No samples")

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return dict(samples=len(values), median_ms=statistics.median(values),
                mean_ms=statistics.mean(values),
                stdev_ms=statistics.stdev(values) if len(values) > 1 else 0.0,
                min_ms=values[0], max_ms=values[-1], q25_ms=percentile(.25),
                q75_ms=percentile(.75), iqr_ms=percentile(.75) - percentile(.25))


def micro_summary(output, meta):
    baseline_name = meta.get("baseline", BASELINE)
    with (output / "micro.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    variants = [v["name"] for v in meta["variants"]]
    cases = {row[0]: row[1:] for row in meta["cases"]}
    expected = {(case, variant, r) for case in cases for variant in variants
                for r in range(1, meta["repeats"] + 1)}
    seen, grouped, order = set(), defaultdict(list), defaultdict(set)
    for row in raw:
        case, variant, repeat = row["case"], row["variant"], int(row["repeat"])
        key = (case, variant, repeat)
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected/duplicate micro sample: {key}")
        seen.add(key)
        if tuple(int(row[k]) for k in CASE_FIELDS[1:]) != tuple(cases[case]):
            raise ValueError(f"Wrong dimensions/epilogue for {key}")
        if (int(row["wrong"]) or int(row["seed"]) != meta["seed"]
                or int(row["iterations"]) != meta["iterations"]
                or int(row["warmup"]) != meta["warmup"]):
            raise ValueError(f"Wrong correctness/configuration for {key}")
        number(row["ms"], positive=True)
        number(row["gflops"])
        number(row["max_abs_error"])
        M, N, K, _, _ = cases[case]
        expected_ref = "CPU_double" if M * N * K <= 2e7 else "cuBLAS_FP32"
        if row["reference"] != expected_ref:
            raise ValueError("Unexpected reference type")
        position = int(row["order"])
        if position not in range(len(variants)) or position in order[case, repeat]:
            raise ValueError("Invalid micro execution order")
        order[case, repeat].add(position)
        grouped[case, variant].append(row)
    if seen != expected:
        raise ValueError(f"Incomplete microbenchmark: {len(expected - seen)} missing samples")
    summary = []
    for case in cases:
        baseline = {int(r["repeat"]): float(r["ms"]) for r in grouped[case, baseline_name]}
        for variant in variants:
            samples = grouped[case, variant]
            summary.append(dict(
                case=case, variant=variant, **dict(zip(CASE_FIELDS[1:], cases[case])),
                **stats([float(r["ms"]) for r in samples]),
                paired_speedup_median=statistics.median(
                    baseline[int(r["repeat"])] / float(r["ms"]) for r in samples),
                gflops_median=statistics.median(float(r["gflops"]) for r in samples),
                max_abs_error=max(float(r["max_abs_error"]) for r in samples),
                reference=samples[0]["reference"],
            ))
    return summary


def model_summaries(output, meta):
    baseline_name = meta.get("baseline", BASELINE)
    names = [v["name"] for v in meta["variants"]] + ["torch"]
    samples, profiles, shapes = [], [], []
    reference = None
    expected_text = None
    for block in range(1, meta["model_blocks"] + 1):
        data = json.loads((output / "model" / f"block{block}.json").read_text())
        if (data["status"] != "complete" or data["block"] != block or data["variants"] != names
                or data["seed"] != meta["seed"]):
            raise ValueError("Incomplete/unexpected model block")
        if reference is None:
            reference, expected_text = data["reference_tokens"], data["reference_transcript"]
        if data["reference_tokens"] != reference or normalize(data["reference_transcript"]) != normalize(expected_text):
            raise ValueError("Reference changed across model blocks")
        expected = {(name, r) for name in names for r in range(1, meta["model_runs"] + 1)}
        seen, order = set(), defaultdict(set)
        for row in data["samples"]:
            key = (row["variant"], row["round"])
            if key not in expected or key in seen or row["block"] != block:
                raise ValueError("Duplicate/unexpected model sample")
            seen.add(key)
            if row["order"] not in range(len(names)) or row["order"] in order[row["round"]]:
                raise ValueError("Invalid model execution order")
            order[row["round"]].add(row["order"])
            number(row["wall_ms"], positive=True)
            number(row["peak_allocated_bytes"], positive=True)
            for field in ("cpu_user_ms", "cpu_system_ms", "voluntary_context_switches", "involuntary_context_switches"):
                number(row[field])
            validate_output(row["token_ids"], row["selected_probs"], row["transcript"], reference, expected_text)
            if row["tokens"] != len(reference):
                raise ValueError("Wrong token count")
            samples.append(dict(row))
        if seen != expected:
            raise ValueError("Incomplete model sample coverage")

        op_keys, op_counts = set(), {}
        for row in data["operation_profiles"]:
            key = (row["variant"], row["operation"])
            if row["block"] != block or row["variant"] not in names or key in op_keys:
                raise ValueError("Invalid/duplicate operation profile")
            op_keys.add(key)
            number(row["gpu_event_ms"], positive=True)
            if not isinstance(row["calls"], int) or row["calls"] < 1:
                raise ValueError("Invalid profile call count")
            op_counts[key] = row["calls"]
            profiles.append(row)
        shape_keys, counts, shape_call_maps = set(), defaultdict(int), defaultdict(dict)
        for row in data["shape_profiles"]:
            shape = tuple(row[k] for k in ("phase", "M", "N", "K", "bias", "act"))
            key = (row["variant"], shape)
            if row["variant"] not in names or row["block"] != block or key in shape_keys:
                raise ValueError("Invalid/duplicate shape profile")
            if row["phase"] not in ("encoder", "decoder", "lm_head", "other"):
                raise ValueError("Invalid profile phase")
            if any(not isinstance(row[k], int) or row[k] < 1 for k in ("M", "N", "K", "calls")):
                raise ValueError("Invalid shape dimensions/calls")
            if row["bias"] not in (0, 1) or row["act"] not in (0, 1):
                raise ValueError("Invalid shape epilogue")
            number(row["gpu_event_ms"], positive=True)
            shape_keys.add(key)
            counts[row["variant"]] += row["calls"]
            shape_call_maps[row["variant"]][shape] = row["calls"]
            shapes.append(row)
        for name in names:
            if counts[name] != op_counts.get((name, "linear"), -1):
                raise ValueError("Shape profile does not cover all linear calls")
            if shape_call_maps[name] != shape_call_maps[baseline_name]:
                raise ValueError("Linear workload differs across variants")

    by_pair = {(r["block"], r["round"], r["variant"]): r["wall_ms"] for r in samples}
    block_summary, pooled = [], []
    for name in names:
        block_ratios = []
        for block in range(1, meta["model_blocks"] + 1):
            rows = [r for r in samples if r["block"] == block and r["variant"] == name]
            ratios = [by_pair[block, r["round"], baseline_name] / r["wall_ms"] for r in rows]
            ratio = statistics.median(ratios)
            block_ratios.append(ratio)
            median = statistics.median(r["wall_ms"] for r in rows)
            for row in rows:
                row["over_2x_block_median"] = row["wall_ms"] > 2 * median
                row["over_5pct_block_median"] = row["wall_ms"] > 1.05 * median
            block_summary.append(dict(block=block, variant=name, **stats([r["wall_ms"] for r in rows]),
                                      paired_speedup_median=ratio))
        rows = [r for r in samples if r["variant"] == name]
        pooled.append(dict(variant=name, blocks=meta["model_blocks"], **stats([r["wall_ms"] for r in rows]),
                           median_block_paired_speedup=statistics.median(block_ratios),
                           min_block_paired_speedup=min(block_ratios), max_block_paired_speedup=max(block_ratios),
                           slow_samples_over_2x=sum(r["over_2x_block_median"] for r in rows),
                           slow_samples_over_5pct=sum(r["over_5pct_block_median"] for r in rows),
                           peak_gpu_gib=max(r["peak_allocated_bytes"] for r in rows) / 2**30))

    op_groups, shape_groups = defaultdict(list), defaultdict(list)
    for row in profiles:
        op_groups[row["variant"], row["operation"]].append(row)
    op_summary = []
    for (name, operation), rows in op_groups.items():
        if len(rows) != meta["model_blocks"] or len({r["calls"] for r in rows}) != 1:
            raise ValueError("Operation workload changed across blocks")
        op_summary.append(dict(variant=name, operation=operation, blocks=len(rows),
                               calls=rows[0]["calls"], median_gpu_event_ms=statistics.median(r["gpu_event_ms"] for r in rows)))
    for row in shapes:
        key = tuple(row[k] for k in ("variant", "phase", "M", "N", "K", "bias", "act"))
        shape_groups[key].append(row)
    shape_summary = []
    for key, rows in shape_groups.items():
        if len(rows) != meta["model_blocks"] or len({r["calls"] for r in rows}) != 1:
            raise ValueError("Shape workload changed across blocks")
        name, *shape = key
        baseline = {r["block"]: r["gpu_event_ms"] for r in shape_groups[(baseline_name, *shape)]}
        total = statistics.median(r["gpu_event_ms"] for r in rows)
        shape_summary.append(dict(zip(("variant", "phase", "M", "N", "K", "bias", "act"), key),
                                  blocks=len(rows), calls_per_inference=rows[0]["calls"],
                                  median_total_ms=total, median_call_ms=total / rows[0]["calls"],
                                  paired_profile_speedup=statistics.median(
                                      baseline[r["block"]] / r["gpu_event_ms"] for r in rows)))
    return samples, pooled, block_summary, op_summary, sorted(shape_summary, key=lambda r: -r["median_total_ms"])


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


OUTPUTS = ("summary.md", "micro_summary.csv", "model_samples.csv", "model_summary.csv",
           "model_blocks.csv", "operation_profiles.csv", "linear_shapes.csv")


def summarize(output):
    for name in OUTPUTS:
        (output / name).unlink(missing_ok=True)
    meta = json.loads((output / "metadata.json").read_text())
    names = [v["name"] for v in meta["variants"]]
    if not names or names[0] != BASELINE or len(set(names)) != len(names):
        raise ValueError("Unique tiled16 baseline must be first")
    if meta["status"] not in ("running", "complete"):
        raise ValueError("Cannot publish a success summary for a failed run")
    micro = micro_summary(output, meta)
    model = None if meta["skip_model"] else model_summaries(output, meta)
    write_csv(output / "micro_summary.csv", micro)
    lines = ["# Task 3: M>1 GEMM experiment", "",
             "Configured data coverage/correctness checks passed. All variants fix M=1 to four warps.", "",
             "## Representative GEMM cases", "",
             "| Case | Variant | Median ms | Paired speedup | GFLOP/s | Max error |",
             "|---|---|---:|---:|---:|---:|"]
    for row in micro:
        if row["case"] in ("decoder_q64", "decoder_gate64", "encoder_fc1", "encoder_fc2", "m1_lm_head_control"):
            lines.append(f"| {row['case']} | {row['variant']} | {row['median_ms']:.6f} | "
                         f"{row['paired_speedup_median']:.3f}x | {row['gflops_median']:.1f} | {row['max_abs_error']:.2e} |")
    if model:
        samples, pooled, blocks, operations, shapes = model
        for name, rows in zip(OUTPUTS[2:], (samples, pooled, blocks, operations, shapes)):
            write_csv(output / name, rows)
        lines += ["", "## Full ASR: all timed samples retained", "",
                  "| Variant | Median ms | Mean ms | IQR ms | Max ms | Median block paired speedup | Slow samples >2x |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in pooled:
            lines.append(f"| {row['variant']} | {row['median_ms']:.2f} | {row['mean_ms']:.2f} | "
                         f"{row['iqr_ms']:.2f} | {row['max_ms']:.2f} | "
                         f"{row['median_block_paired_speedup']:.3f}x | {row['slow_samples_over_2x']} |")
        lines += ["", "Independent blocks are separate model processes in one GPU allocation.",
                  "Within a block, variants share fixed weights and run in shuffled order each round.",
                  "Speedups pair each round against tiled16, then report the median of block medians.",
                  "Exact token IDs and expected normalized transcription are checked on every timed run.",
                  "One supplied audio is evaluated; this does not establish dataset-wide WER/CER.", "",
                  "## Highest-cost baseline linear shapes", "",
                  "| Phase | M,N,K | Bias/GELU | Calls | Profile total ms |",
                  "|---|---|---|---:|---:|"]
        for row in [r for r in shapes if r["variant"] == BASELINE][:12]:
            lines.append(f"| {row['phase']} | {row['M']},{row['N']},{row['K']} | "
                         f"{row['bias']}/{row['act']} | {row['calls_per_inference']} | {row['median_total_ms']:.3f} |")
        lines += ["", "Shape/operation profiles are separate diagnostic generations, excluded from normal timings.",
                  "GPU-event spans include allocation/launch glue and profiler overhead; pure kernel timing is in micro.csv.",
                  "CPU/context-switch counters and UTC timestamps are in model_samples.csv. They do not alone identify a stall cause."]
    else:
        lines += ["", "Full-model evaluation was explicitly skipped; no ASR latency/token-consistency claim is established."]
    lines += ["", "No kernel default is changed automatically. Compare blocks and shapes before choosing a configuration.",
              "Hardware telemetry is best-effort and refers to physical GPUs, not per-MIG utilization."]
    report = output / "summary.md"
    report.write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        print(summarize(args.output.resolve()))
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        parser.exit(1, f"Summary failed: {error}\n")
