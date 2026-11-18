#!/usr/bin/env python3
"""Validate both microbenchmarks and paired-model/shape profiles before publishing."""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "task3_gemm"))
import results as shared
from attention_spec import ATTENTION_FIELDS


def attention_micro_summary(output, meta):
    cases = {row[0]: row[1:] for row in meta["attention_cases"]}
    names = [v["name"] for v in meta["variants"]]
    expected = {(c, n, r) for c in cases for n in names for r in range(1, meta["repeats"] + 1)}
    with (output / "attention_micro.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen, groups, order = set(), defaultdict(list), defaultdict(set)
    for row in rows:
        c, name, repeat = row["case"], row["variant"], int(row["repeat"])
        key = (c, name, repeat)
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected/duplicate attention sample: {key}")
        seen.add(key)
        values = tuple(float(row[f]) if f == "q_range" else int(row[f]) for f in ATTENTION_FIELDS[1:])
        if values != tuple(cases[c]):
            raise ValueError(f"Attention dimensions/pattern differ: {key}")
        if (int(row["wrong"]) or int(row["seed"]) != meta["seed"] or int(row["warmup"]) != meta["warmup"]
                or int(row["iterations"]) != meta["iterations"]):
            raise ValueError("Incorrect attention result/configuration")
        B, H, _, Sq, Sk, D, _, _, _ = cases[c]
        expected_ref = "CPU_double" if B * H * Sq * Sk * D <= 2e7 else "cuBLAS_CPU_softmax"
        if row["reference"] != expected_ref:
            raise ValueError("Wrong attention reference type")
        shared.number(row["ms"], positive=True)
        shared.number(row["gflops"])
        shared.number(row["max_abs_error"])
        position = int(row["order"])
        if position not in range(len(names)) or position in order[c, repeat]:
            raise ValueError("Invalid attention execution order")
        order[c, repeat].add(position)
        groups[c, name].append(row)
    if seen != expected:
        raise ValueError("Incomplete attention microbenchmark coverage")
    summary = []
    for c in cases:
        base = {int(r["repeat"]): float(r["ms"]) for r in groups[c, meta["baseline"]]}
        for name in names:
            rr = groups[c, name]
            summary.append(dict(case=c, variant=name, **dict(zip(ATTENTION_FIELDS[1:], cases[c])),
                                **shared.stats([float(r["ms"]) for r in rr]),
                                paired_speedup_median=statistics.median(base[int(r["repeat"])] / float(r["ms"]) for r in rr),
                                max_abs_error=max(float(r["max_abs_error"]) for r in rr), reference=rr[0]["reference"]))
    return summary


def validate_attention_profiles(output, meta):
    names = [v["name"] for v in meta["variants"]] + ["torch"]
    groups, checks = defaultdict(list), []
    fields = ("phase", *ATTENTION_FIELDS[1:8])  # no synthetic pattern/range in real-model shapes
    first_workload = None
    for block in range(1, meta["model_blocks"] + 1):
        data = json.loads((output / "model" / f"block{block}.json").read_text())
        if data.get("profile_schema") != 2:
            raise ValueError("Expected original-event profile schema v2")
        sums, counts, workload, seen = defaultdict(float), defaultdict(int), defaultdict(dict), set()
        for row in data["attention_profiles"]:
            name = row["variant"]
            key = tuple(row[f] for f in fields)
            if name not in names or row["block"] != block or (name, key) in seen:
                raise ValueError("Unexpected/duplicate attention profile")
            seen.add((name, key))
            if row["phase"] not in ("encoder", "decoder", "other"):
                raise ValueError("Invalid attention phase")
            if any(not isinstance(row[f], int) or row[f] < 1 for f in ATTENTION_FIELDS[1:7]):
                raise ValueError("Invalid attention shape")
            if (row["causal"] not in (0, 1) or row["num_heads"] % row["num_kv_heads"]
                    or row["head_dim"] not in (32, 64, 96, 128)
                    or not isinstance(row["calls"], int) or row["calls"] < 1):
                raise ValueError("Invalid attention shape/calls")
            shared.number(row["gpu_event_ms"], positive=True)
            sums[name] += row["gpu_event_ms"]
            counts[name] += row["calls"]
            workload[name][key] = row["calls"]
            groups[(name, *key)].append(row)
        baseline_workload = workload[meta["baseline"]]
        if not baseline_workload or any(workload[n] != baseline_workload for n in names):
            raise ValueError("Attention workload differs across variants")
        if first_workload is None:
            first_workload = baseline_workload
        if baseline_workload != first_workload:
            raise ValueError("Attention workload changed across blocks")
        for name in names:
            for op, rows in (("linear", data["shape_profiles"]), ("attention", data["attention_profiles"])):
                original = [r for r in data["operation_profiles"] if r["variant"] == name and r["operation"] == op]
                if len(original) != 1:
                    raise ValueError("Missing/duplicate original operation record")
                grouped = [r for r in rows if r["variant"] == name]
                total, calls = sum(r["gpu_event_ms"] for r in grouped), sum(r["calls"] for r in grouped)
                actual = original[0]
                if calls != actual["calls"] or not math.isclose(total, actual["gpu_event_ms"], rel_tol=1e-8, abs_tol=1e-5):
                    raise ValueError(f"{op} shape events do not match original events")
                checks.append(dict(block=block, variant=name, operation=op, calls=calls,
                                   original_ms=actual["gpu_event_ms"], shape_sum_ms=total,
                                   difference_ms=total - actual["gpu_event_ms"]))
    summary = []
    for key, rows in groups.items():
        if len(rows) != meta["model_blocks"]:
            raise ValueError("Incomplete attention shape coverage")
        name, *shape = key
        base = {r["block"]: r["gpu_event_ms"] for r in groups[(meta["baseline"], *shape)]}
        total = statistics.median(r["gpu_event_ms"] for r in rows)
        summary.append(dict(zip(("variant", *fields), key), blocks=len(rows), calls_per_inference=rows[0]["calls"],
                            median_total_ms=total, median_call_ms=total / rows[0]["calls"],
                            paired_profile_speedup=statistics.median(base[r["block"]] / r["gpu_event_ms"] for r in rows)))
    return sorted(summary, key=lambda r: -r["median_total_ms"]), checks


def factorial_effects(samples, meta):
    if meta["stage"] != "factorial" or not {"base", "hybrid", "attention", "combined"}.issubset(
            {v["name"] for v in meta["variants"]}):
        return []
    by = {(r["block"], r["round"], r["variant"]): r["wall_ms"] for r in samples}
    rows = []
    for block in range(1, meta["model_blocks"] + 1):
        effects = defaultdict(list)
        for repeat in range(1, meta["model_runs"] + 1):
            a, b, c, d = [by[block, repeat, n] for n in ("base", "hybrid", "attention", "combined")]
            for key, value in dict(hybrid_saved_ms=a-b, attention_saved_ms=a-c, combined_saved_ms=a-d,
                                   interaction_ms=d-b-c+a).items():
                effects[key].append(value)
        rows.append(dict(block=block, **{k: statistics.median(v) for k, v in effects.items()}))
    return rows


OUTPUTS = ("summary.md", "micro_summary.csv", "attention_micro_summary.csv", "model_samples.csv",
           "model_summary.csv", "model_blocks.csv", "operation_profiles.csv", "linear_shapes.csv",
           "attention_shapes.csv", "profile_boundary_checks.csv", "factorial_effects.csv")


def summarize(output):
    for name in OUTPUTS:
        (output / name).unlink(missing_ok=True)
    meta = json.loads((output / "metadata.json").read_text())
    if meta.get("stage") not in ("sweep", "factorial"):
        raise ValueError("Not an attention experiment directory")
    names = [v["name"] for v in meta["variants"]]
    if (not names or names[0] != meta["baseline"] or len(set(names)) != len(names)
            or meta["status"] not in ("running", "complete")):
        raise ValueError("Invalid baseline/status")
    att = attention_micro_summary(output, meta)
    gemm = shared.micro_summary(output, meta)
    model = None if meta["skip_model"] else shared.model_summaries(output, meta)
    attention, checks = (None, None) if model is None else validate_attention_profiles(output, meta)
    shared.write_csv(output / "micro_summary.csv", gemm)
    shared.write_csv(output / "attention_micro_summary.csv", att)
    lines = [f"# Attention experiment: {meta['stage']}", "", f"Baseline: {meta['baseline']}; M=1/w4 fixed; GEMM=rb4x4 with optional narrow-N rule.",
             "", "Configured correctness/coverage checks passed. No kernel default is automatically changed.", "",
             "## Attention microbenchmarks", "", "| Case | Variant | Median ms | Paired speedup | Max error | Reference |",
             "|---|---|---:|---:|---:|---|"]
    for row in att:
        if row["case"] in ("encoder", "decoder", "masked_prefix"):
            lines.append(f"| {row['case']} | {row['variant']} | {row['median_ms']:.6f} | "
                         f"{row['paired_speedup_median']:.3f}x | {row['max_abs_error']:.2e} | {row['reference']} |")
    if model:
        samples, pooled, blocks, operations, linear = model
        effects = factorial_effects(samples, meta)
        for name, rows in (("model_samples.csv", samples), ("model_summary.csv", pooled),
                           ("model_blocks.csv", blocks), ("operation_profiles.csv", operations),
                           ("linear_shapes.csv", linear), ("attention_shapes.csv", attention),
                           ("profile_boundary_checks.csv", checks)):
            shared.write_csv(output / name, rows)
        if effects:
            shared.write_csv(output / "factorial_effects.csv", effects)
        lines += ["", "## ASR: all samples retained", "",
                  "| Variant | Median ms | Mean ms | IQR ms | Max ms | Block-paired speedup | >5% samples | >2x samples |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in pooled:
            lines.append(f"| {row['variant']} | {row['median_ms']:.2f} | {row['mean_ms']:.2f} | {row['iqr_ms']:.2f} | "
                         f"{row['max_ms']:.2f} | {row['median_block_paired_speedup']:.3f}x | "
                         f"{row['slow_samples_over_5pct']} | {row['slow_samples_over_2x']} |")
        lines += ["", "## Baseline attention by phase", "",
                  "| Phase | Q/K sequence | Heads/KV heads | Dim | Calls | Profile ms |", "|---|---|---|---:|---:|---:|"]
        for row in attention:
            if row["variant"] == meta["baseline"]:
                lines.append(f"| {row['phase']} | {row['seq_q']}/{row['seq_k']} | {row['num_heads']}/{row['num_kv_heads']} | "
                             f"{row['head_dim']} | {row['calls_per_inference']} | {row['median_total_ms']:.3f} |")
        if effects:
            lines += ["", "## Paired factorial effects (medians across block medians)", ""]
            for field in ("hybrid_saved_ms", "attention_saved_ms", "combined_saved_ms", "interaction_ms"):
                lines.append(f"- {field}: {statistics.median(r[field] for r in effects):.3f} ms")
            lines += ["", "interaction_ms = combined - hybrid - attention + base; positive means slower than additive prediction."]
        lines += ["", "Linear/attention shapes reuse the original ops events; sums and call counts were checked against operation totals.",
                  "Profiles are separate diagnostic generations. Allocations/launch glue remain inside the original event boundary.",
                  "Every timed generation checks exact token IDs and expected transcription. One supplied audio is not a WER/CER dataset.",
                  "Blocks are independent model processes in one GPU allocation; pairs match each timing round against the stage baseline."]
    else:
        lines += ["", "Model evaluation was skipped; no ASR speedup or token-consistency claim is established."]
    lines += ["", "All raw slow samples are retained. >5% and >2x are diagnostic flags, never exclusion rules.",
              "Use sweep results to choose a warp count explicitly, then confirm it with the factorial stage."]
    report = output / "summary.md"
    report.write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    try:
        print(summarize(args.output.resolve()))
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        p.exit(1, f"Summary failed: {error}\n")
