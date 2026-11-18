"""Offline fixtures validate experiment tools, never GPU correctness/performance."""
import contextlib
import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_attention as runner
from attention_spec import ATTENTION_CASES, ATTENTION_FIELDS, WARPS, configs, variant_flags
from attention_results import (attention_micro_summary, factorial_effects, summarize,
                               validate_attention_profiles)
from results import model_summaries, write_csv


def metadata(stage="sweep", skip_model=True):
    variants = configs(stage)
    if stage == "sweep":
        variants = variants[:2]
    return dict(status="running", stage=stage, baseline=variants[0]["name"], variants=variants,
                attention_cases=ATTENTION_CASES, cases=runner.gemm_cases(stage), seed=21,
                warmup=5, iterations=20, repeats=2, skip_model=skip_model,
                model_runs=3, model_blocks=2)


def attention_fixture(meta):
    rows = []
    for case in meta["attention_cases"]:
        _, b, h, _, sq, sk, d, *_ = case
        for repeat in range(1, meta["repeats"] + 1):
            for position, variant in enumerate(meta["variants"]):
                ms = 10 / (position + 1)
                rows.append(dict(zip(ATTENTION_FIELDS, case), variant=variant["name"],
                                 repeat=repeat, order=position, ms=ms, gflops=1, max_abs_error=1e-6,
                                 wrong=0, reference="CPU_double" if b*h*sq*sk*d <= 2e7 else "cuBLAS_CPU_softmax",
                                 seed=meta["seed"], warmup=meta["warmup"], iterations=meta["iterations"]))
    return rows


def gemm_fixture(meta):
    rows = []
    for case, m, n, k, bias, act in meta["cases"]:
        for repeat in range(1, meta["repeats"] + 1):
            for order, variant in enumerate(meta["variants"]):
                rows.append(dict(case=case, M=m, N=n, K=k, bias=bias, act=act,
                                 variant=variant["name"], repeat=repeat, order=order, ms=1,
                                 gflops=2*m*n*k/1e6, max_abs_error=1e-6, wrong=0,
                                 reference="CPU_double" if m*n*k <= 2e7 else "cuBLAS_FP32",
                                 seed=meta["seed"], warmup=meta["warmup"], iterations=meta["iterations"]))
    return rows


def block_fixture(meta, block):
    names = [v["name"] for v in meta["variants"]] + ["torch"]
    times = dict(base=100, hybrid=99, attention=80, combined=78, torch=50)
    samples, operations, linear, attention = [], [], [], []
    for repeat in range(1, meta["model_runs"] + 1):
        for order, name in enumerate(names):
            samples.append(dict(block=block, round=repeat, variant=name, order=order,
                                wall_ms=times.get(name, 100), tokens=2, token_ids=[1, 2],
                                selected_probs=[.9, .8], transcript="Fixture answer.",
                                started_utc="2026-11-17T21:00:00+00:00",
                                finished_utc="2026-11-17T21:00:01+00:00", peak_allocated_bytes=1024,
                                cpu_user_ms=1, cpu_system_ms=0, voluntary_context_switches=1,
                                involuntary_context_switches=0))
    for name in names:
        operations.extend([
            dict(block=block, variant=name, operation="linear", backend="cuda", calls=3, gpu_event_ms=60),
            dict(block=block, variant=name, operation="attention", backend="cuda", calls=4, gpu_event_ms=30),
        ])
        linear.append(dict(block=block, variant=name, phase="decoder", M=64, N=512, K=2048,
                           bias=0, act=0, calls=3, gpu_event_ms=60, mean_ms=20))
        for phase, h, kv, sq, sk, d, causal, calls, ms in (
                ("encoder", 20, 20, 1500, 1500, 64, 0, 1, 24),
                ("decoder", 16, 4, 64, 64, 128, 1, 3, 6)):
            attention.append(dict(block=block, variant=name, phase=phase, batch=1, num_heads=h,
                                  num_kv_heads=kv, seq_q=sq, seq_k=sk, head_dim=d, causal=causal,
                                  calls=calls, gpu_event_ms=ms, mean_ms=ms/calls))
    return dict(status="complete", block=block, seed=meta["seed"], variants=names,
                profile_schema=2, reference_tokens=[1, 2], reference_transcript="Fixture answer",
                samples=samples, operation_profiles=operations, shape_profiles=linear,
                attention_profiles=attention)


class AttentionExperimentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def prepare(self, stage="sweep", skip=True):
        meta = metadata(stage, skip)
        (self.output / "metadata.json").write_text(json.dumps(meta))
        write_csv(self.output / "attention_micro.csv", attention_fixture(meta))
        write_csv(self.output / "micro.csv", gemm_fixture(meta))
        if not skip:
            (self.output / "model").mkdir(exist_ok=True)
            for block in range(1, meta["model_blocks"] + 1):
                self.save_block(block_fixture(meta, block))
        return meta

    def save_block(self, data):
        (self.output / "model" / f"block{data['block']}.json").write_text(json.dumps(data))

    def test_sweep_is_full_warps_by_softmax_factorial(self):
        variants = configs("sweep")
        self.assertEqual(len(variants), 10)
        self.assertEqual(variants[0]["name"], "w4_standard")
        self.assertEqual({(v["warps"], v["online"]) for v in variants},
                         {(w, mode) for w in WARPS for mode in (0, 1)})
        for v in variants + configs("factorial", 8):
            flags = variant_flags(v)
            for flag in ("-DASR_TASK3_M1_ENABLED=1", "-DASR_TASK3_M1_WARPS=4",
                         "-DASR_TASK3_GEMM_IMPL=1", "-DASR_TASK3_RM=4", "-DASR_TASK3_RN=4"):
                self.assertIn(flag, flags)

    def test_confirmation_changes_only_selected_factors(self):
        for online in (0, 1):
            variants = configs("factorial", 8, online)
            self.assertEqual([(v["warps"], v["online"], v["hybrid"]) for v in variants],
                             [(4, 0, 0), (4, 0, 1), (8, online, 0), (8, online, 1)])
        with self.assertRaises(ValueError):
            configs("typo")

    def test_stage_model_defaults_and_selection(self):
        self.assertTrue(runner.parse_args([]).skip_model)
        args = runner.parse_args(["--stage", "factorial", "--selected-warps", "8", "--selected-online", "standard"])
        self.assertFalse(args.skip_model)
        self.assertEqual(args.selected_online, 0)
        self.assertTrue(runner.parse_args(["--stage", "factorial", "--no-model"]).skip_model)
        self.assertFalse(runner.parse_args(["--model"]).skip_model)

    def test_invalid_arguments(self):
        for args in (["--variants", "w4_direct"], ["--variants", "w4_standard", "w4_standard"],
                     ["--dry-run"], ["--sm", "90a"], ["--repeats", "0"],
                     ["--selected-warps", "3"], ["--stage", "factorial", "--variants", "w4_standard"],
                     ["--model", "--no-model"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(args)

    def test_dry_run_is_side_effect_free_and_frozen(self):
        for stage in ("sweep", "factorial"):
            target = self.output / stage
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                code = runner.main(["--dry-run", "--sm", "90", "--stage", stage,
                                    "--output", str(target), "--sanitizer"])
            text = stream.getvalue()
            self.assertEqual(code, 0)
            self.assertFalse(target.exists())
            self.assertIn("source/experiment/benchmark_attention.cu", text)
            self.assertIn("source/experiment/benchmark_gemm.cu", text)
            self.assertIn("nvcc_probe.log", text)
            for tool in ("memcheck", "racecheck", "synccheck"):
                self.assertEqual(text.count(f"--tool {tool}"), 2)
            self.assertEqual("block3.json" in text, stage == "factorial")

    def test_hybrid_holdouts_include_unseen_m_and_controls(self):
        cases = {c[0]: c[1:] for c in runner.gemm_cases("factorial")}
        for m in (32, 48, 63, 65, 80, 96):
            self.assertEqual(cases[f"kv_m{m}"], (m, 512, 2048, 0, 0))
        self.assertEqual(cases["kv_bias_gelu"], (64, 512, 2048, 1, 1))
        self.assertEqual(len(cases), 33)
        self.assertEqual(len(runner.gemm_cases("sweep")), 4)

    def test_attention_micro_summary_and_masked_case_coverage(self):
        meta = self.prepare()
        rows = attention_micro_summary(self.output, meta)
        self.assertEqual(len(rows), 28)
        row = next(r for r in rows if r["case"] == "encoder" and r["variant"] != meta["baseline"])
        self.assertEqual(row["paired_speedup_median"], 2)
        self.assertEqual(row["reference"], "cuBLAS_CPU_softmax")
        self.assertTrue(any(r["case"] == "masked_prefix" for r in rows))

    def test_missing_and_duplicate_micro_samples_rejected(self):
        meta = self.prepare()
        rows = attention_fixture(meta)
        for changed in (rows[:-1], rows + rows[:1]):
            write_csv(self.output / "attention_micro.csv", changed)
            with self.assertRaises(ValueError):
                attention_micro_summary(self.output, meta)

    def test_invalid_micro_dimensions_errors_reference_and_order(self):
        meta = self.prepare()
        for field, value in (("wrong", 1), ("ms", "nan"), ("ms", 0), ("seq_k", 99),
                             ("order", 99), ("reference", "unverified"), ("seed", 999),
                             ("q_range", 10), ("pattern", 99), ("max_abs_error", "inf")):
            rows = attention_fixture(meta)
            rows[0][field] = value
            write_csv(self.output / "attention_micro.csv", rows)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                attention_micro_summary(self.output, meta)

    def test_full_summary_profiles_and_paired_factorial_effects(self):
        meta = self.prepare("factorial", skip=False)
        report = summarize(self.output).read_text()
        self.assertIn("Baseline: base", report)
        self.assertIn("interaction_ms", report)
        samples, pooled, *_ = model_summaries(self.output, meta)
        effects = factorial_effects(samples, meta)
        self.assertEqual(effects[0], dict(block=1, hybrid_saved_ms=1, attention_saved_ms=20,
                                         combined_saved_ms=22, interaction_ms=-1))
        self.assertEqual(next(r for r in pooled if r["variant"] == "combined")["median_block_paired_speedup"], 100/78)
        with (self.output / "profile_boundary_checks.csv").open() as handle:
            checks = list(csv.DictReader(handle))
        self.assertEqual(len(checks), 20)
        self.assertTrue(all(float(r["difference_ms"]) == 0 for r in checks))
        self.assertTrue((self.output / "attention_shapes.csv").is_file())

    def test_model_skipped_does_not_claim_asr_results(self):
        self.prepare()
        self.assertIn("Model evaluation was skipped", summarize(self.output).read_text())
        self.assertFalse((self.output / "model_summary.csv").exists())

    def test_original_event_boundary_mismatch_rejected_for_both_ops(self):
        meta = self.prepare("factorial", skip=False)
        for field in ("shape_profiles", "attention_profiles"):
            data = block_fixture(meta, 1)
            data[field][0]["gpu_event_ms"] += .5
            self.save_block(data)
            with self.assertRaisesRegex(ValueError, "shape events do not match"):
                validate_attention_profiles(self.output, meta)

    def test_attention_workload_schema_and_calls_rejected(self):
        meta = self.prepare("factorial", skip=False)
        for change in ("missing", "duplicate", "phase", "calls", "schema", "heads", "nan"):
            data = block_fixture(meta, 1)
            row = data["attention_profiles"][0]
            if change == "missing": data["attention_profiles"].pop()
            elif change == "duplicate": data["attention_profiles"].append(dict(row))
            elif change == "phase": row["phase"] = "unknown"
            elif change == "calls": row["calls"] = 1.5
            elif change == "schema": data["profile_schema"] = 1
            elif change == "heads": row["num_kv_heads"] = 3
            else: row["gpu_event_ms"] = float("nan")
            self.save_block(data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_attention_profiles(self.output, meta)

    def test_token_ids_must_match_even_with_matching_text(self):
        meta = self.prepare("factorial", skip=False)
        data = block_fixture(meta, 1)
        data["samples"][0]["token_ids"] = [2, 1]
        self.save_block(data)
        with self.assertRaisesRegex(ValueError, "token IDs"):
            summarize(self.output)

    def test_five_percent_tail_retained_below_two_times_median(self):
        meta = self.prepare("factorial", skip=False)
        data = block_fixture(meta, 1)
        next(r for r in data["samples"] if r["variant"] == "attention")["wall_ms"] = 88
        self.save_block(data)
        samples, pooled, *_ = model_summaries(self.output, meta)
        row = next(r for r in pooled if r["variant"] == "attention")
        self.assertEqual(row["slow_samples_over_5pct"], 1)
        self.assertEqual(row["slow_samples_over_2x"], 0)
        self.assertEqual(row["samples"], 6)
        self.assertEqual(row["max_ms"], 88)
        self.assertEqual(len(samples), 30)

    def test_bad_resummary_removes_stale_success(self):
        meta = self.prepare()
        summarize(self.output)
        write_csv(self.output / "attention_micro.csv", attention_fixture(meta)[:-1])
        with self.assertRaises(ValueError):
            summarize(self.output)
        self.assertFalse((self.output / "summary.md").exists())
        self.assertFalse((self.output / "attention_micro_summary.csv").exists())

    def test_offline_lifecycle_snapshot_environment_and_early_failure(self):
        for fail in (False, True):
            target, logs = self.output / ("failed" if fail else "success"), []

            def fake_step(step, output):
                logs.append(step.log)
                if fail and step.log == "preflight/nvcc_probe.log":
                    raise RuntimeError("Offline nvcc probe failure")
                meta = json.loads((output / "metadata.json").read_text())
                if step.log == "attention_micro.log":
                    write_csv(output / "attention_micro.csv", attention_fixture(meta))
                elif step.log == "gemm_micro.log":
                    write_csv(output / "micro.csv", gemm_fixture(meta))

            before = {k: os.environ.get(k) for k in ("TMPDIR", "NVCC_CCBIN")}
            with mock.patch.object(runner.shutil, "which", return_value="/opt/cuda/bin/nvcc"), \
                    mock.patch.object(runner.platform, "platform", return_value="offline"), \
                    mock.patch.object(runner.runtime, "gpu_info", return_value=dict(name="offline fixture", sm="90")), \
                    mock.patch.object(runner.runtime.subprocess, "check_output", return_value="fixture\n"), \
                    mock.patch.object(runner.runtime, "run_step", side_effect=fake_step), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = runner.main(["--output", str(target), "--no-telemetry", "--repeats", "2",
                                    "--variants", "w4_standard", "w1_standard",
                                    "--cuda-tmpdir", str(self.output / "scratch")])
            self.assertEqual(code, int(fail))
            self.assertEqual({k: os.environ.get(k) for k in before}, before)
            meta = json.loads((target / "metadata.json").read_text())
            self.assertEqual(meta["status"], "failed" if fail else "complete")
            self.assertEqual((target / "summary.md").exists(), not fail)
            self.assertEqual((target / "FAILED.txt").exists(), fail)
            if fail:
                self.assertFalse(any(log.startswith("build/") for log in logs))
            for rel, sha in json.loads((target / "source_sha256.json").read_text()).items():
                self.assertEqual(hashlib.sha256((target / "source" / rel).read_bytes()).hexdigest(), sha)
            for file in ("model_runner.py", "spec.py", "benchmark_gemm.cu", "benchmark_attention.cu", "attention_spec.py"):
                self.assertTrue((target / "source" / "experiment" / file).is_file())
            self.assertNotIn(b"\r", (target / "attention_cases.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
