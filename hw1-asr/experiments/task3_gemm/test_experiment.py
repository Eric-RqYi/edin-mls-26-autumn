"""Offline synthetic tool tests. No generated fixture is a GPU performance result."""
import contextlib
import csv
import io
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_experiment as runner
from model_runner import ShapeProfiler, validate_output
from results import micro_summary, model_summaries, summarize, write_csv
from spec import BASELINE, CASES, VARIANTS, variant_flags


def metadata(skip_model=False):
    return dict(status="running", variants=[dict(name=n) for n in (BASELINE, "rb2x2")],
                cases=CASES, repeats=2, warmup=5, iterations=20, seed=21,
                model_runs=3, model_blocks=2, skip_model=skip_model)


def micro_fixture(meta):
    rows = []
    for case, M, N, K, bias, act in meta["cases"]:
        for repeat in range(1, meta["repeats"] + 1):
            for position, variant in enumerate(meta["variants"]):
                ms = 10 if position == 0 or M == 1 else 5
                rows.append(dict(case=case, variant=variant["name"], M=M, N=N, K=K,
                                 bias=bias, act=act, repeat=repeat, order=position, ms=ms,
                                 gflops=2 * M * N * K / (ms * 1e6), max_abs_error=1e-6,
                                 wrong=0, reference="CPU_double" if M * N * K <= 2e7 else "cuBLAS_FP32",
                                 seed=meta["seed"], warmup=meta["warmup"], iterations=meta["iterations"]))
    return rows


def block_fixture(meta, block):
    names = [v["name"] for v in meta["variants"]] + ["torch"]
    rows, profiles, shapes = [], [], []
    for round_id in range(1, meta["model_runs"] + 1):
        for order, name in enumerate(names):
            ms = 10.0 if name == BASELINE else 5.0 if name != "torch" else 2.5
            rows.append(dict(block=block, round=round_id, variant=name, order=order, wall_ms=ms,
                             token_ids=[1, 2], selected_probs=[.9, .8], tokens=2, transcript="Fixture answer.",
                             started_utc="2026-11-17T17:00:00+00:00", finished_utc="2026-11-17T17:00:01+00:00",
                             peak_allocated_bytes=8 * 2**30, cpu_user_ms=1, cpu_system_ms=0,
                             voluntary_context_switches=1, involuntary_context_switches=0))
    for name in names:
        factor = 1 if name == BASELINE else .5
        profiles.extend([
            dict(block=block, variant=name, operation="linear", backend="cuda", calls=6, gpu_event_ms=11 * factor),
            dict(block=block, variant=name, operation="attention", backend="cuda", calls=4, gpu_event_ms=2),
        ])
        for phase, M, N, K, calls, ms in (
                ("encoder", 1500, 1280, 1280, 2, 4),
                ("decoder", 64, 2048, 2048, 3, 6),
                ("lm_head", 1, 59264, 2048, 1, 1)):
            shapes.append(dict(block=block, variant=name, phase=phase, M=M, N=N, K=K, bias=0, act=0,
                               calls=calls, gpu_event_ms=ms * factor, mean_ms=ms * factor / calls))
    return dict(status="complete", block=block, seed=meta["seed"], variants=names,
                reference_tokens=[1, 2], reference_transcript="Fixture answer", samples=rows,
                operation_profiles=profiles, shape_profiles=shapes)


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)

    def prepare(self, skip=False):
        meta = metadata(skip)
        (self.output / "metadata.json").write_text(json.dumps(meta))
        write_csv(self.output / "micro.csv", micro_fixture(meta))
        if not skip:
            (self.output / "model").mkdir(exist_ok=True)
            for block in range(1, meta["model_blocks"] + 1):
                (self.output / "model" / f"block{block}.json").write_text(json.dumps(block_fixture(meta, block)))
        return meta

    def test_all_variants_keep_m1_fixed(self):
        for name in VARIANTS:
            flags = variant_flags(name)
            self.assertIn("-DASR_TASK3_M1_ENABLED=1", flags)
            self.assertIn("-DASR_TASK3_M1_WARPS=4", flags)

    def test_dry_run_is_offline_and_read_only(self):
        target = self.output / "never_created"
        with contextlib.redirect_stdout(io.StringIO()) as text:
            code = runner.main(["--dry-run", "--sm", "90", "--output", str(target), "--sanitizer"])
        self.assertEqual(code, 0)
        self.assertFalse(target.exists())
        self.assertIn("block3.json", text.getvalue())
        self.assertIn("synccheck", text.getvalue())
        self.assertIn("racecheck", text.getvalue())
        self.assertIn("source/csrc", text.getvalue())

    def test_invalid_arguments(self):
        for args in (["--variants", "rb2x2"], ["--variants", BASELINE, BASELINE],
                     ["--dry-run"], ["--sm", "90a"], ["--model-blocks", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(args)

    def test_independent_builds_and_skip_model(self):
        args = runner.parse_args(["--skip-model", "--variants", BASELINE, "rb4x4"])
        variants = runner.make_variants(self.output, args.variants)
        self.assertEqual(len({v["build"] for v in variants}), 2)
        steps = runner.make_steps(args, self.output, "90", variants, Path("/opt/cuda"))
        self.assertFalse(any(s.log.startswith("model/") for s in steps))
        self.assertEqual(sum(s.argv[-1] == "test" for s in steps), 2)

    def test_compiler_preflight_precedes_variant_builds(self):
        args = runner.parse_args(["--skip-model", "--variants", BASELINE])
        variants = runner.make_variants(self.output, args.variants)
        steps = runner.make_steps(args, self.output, "90", variants, Path("/opt/cuda"))
        self.assertEqual([s.log for s in steps[:3]], [
            "preflight/host_compiler.log", "preflight/host_preprocess.log", "preflight/nvcc_probe.log"])
        self.assertEqual(steps[3].log, "build/tiled16.log")
        self.assertIn("--verbose", steps[2].argv)
        self.assertIn("--keep-dir", steps[2].argv)

    def test_private_tmpdir_and_real_host_preprocessing(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("No local C++ compiler; this test does not require CUDA")
        (self.output / "source").mkdir()
        with mock.patch.dict(os.environ, NVCC_CCBIN=compiler, TMPDIR="/nonexistent-scheduler-tmp"):
            settings = runner.prepare_build_environment(self.output, self.output / "scratch")
            self.assertEqual(settings["inherited_tmpdir"], "/nonexistent-scheduler-tmp")
            self.assertTrue(Path(settings["tmpdir"]).is_dir())
            self.assertEqual(os.environ["TMPDIR"], settings["tmpdir"])
            first_tmpdir = settings["tmpdir"]
            steps = runner.preflight_steps(self.output, "90", settings["host_compiler"])
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run_step(steps[0], self.output)
                runner.run_step(steps[1], self.output)
            self.assertTrue((self.output / "preflight" / "host_probe.ii").is_file())
            self.assertIn(first_tmpdir, (self.output / "preflight" / "host_preprocess.log").read_text())

    def test_missing_host_compiler_rejected(self):
        with mock.patch.dict(os.environ, NVCC_CCBIN="/nonexistent/compiler"), self.assertRaisesRegex(
                RuntimeError, "Host compiler is not executable"):
            runner.prepare_build_environment(self.output, self.output / "scratch")

    def test_token_ids_checked_even_if_text_matches(self):
        with self.assertRaisesRegex(ValueError, "token IDs"):
            validate_output([2, 1], [.9, .8], "Fixture answer", [1, 2], "Fixture answer")

    def test_nonfinite_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            validate_output([1], [math.nan], "Fixture answer", [1], "Fixture answer")

    def test_micro_coverage_and_speedup(self):
        meta = self.prepare(skip=True)
        summary = micro_summary(self.output, meta)
        row = next(r for r in summary if r["case"] == "decoder_q64" and r["variant"] == "rb2x2")
        self.assertEqual(row["paired_speedup_median"], 2)

    def test_missing_micro_sample(self):
        meta = self.prepare(skip=True)
        write_csv(self.output / "micro.csv", micro_fixture(meta)[:-1])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            micro_summary(self.output, meta)

    def test_duplicate_micro_sample(self):
        meta = self.prepare(skip=True)
        rows = micro_fixture(meta)
        write_csv(self.output / "micro.csv", rows + rows[:1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            micro_summary(self.output, meta)

    def test_micro_wrong_nan_and_shape_rejected(self):
        meta = self.prepare(skip=True)
        for key, value in (("wrong", 1), ("ms", "nan"), ("M", 999), ("order", 99)):
            rows = micro_fixture(meta)
            rows[0][key] = value
            write_csv(self.output / "micro.csv", rows)
            with self.assertRaises(ValueError):
                micro_summary(self.output, meta)

    def test_full_summary_exports_profiles(self):
        self.prepare()
        path = summarize(self.output)
        self.assertTrue(path.exists())
        with (self.output / "model_summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        row = next(r for r in rows if r["variant"] == "rb2x2")
        self.assertEqual(float(row["median_block_paired_speedup"]), 2)
        self.assertEqual(int(row["samples"]), 6)
        self.assertTrue((self.output / "linear_shapes.csv").exists())

    def test_missing_model_sample_rejected(self):
        meta = self.prepare()
        data = block_fixture(meta, 1)
        data["samples"].pop()
        (self.output / "model" / "block1.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            model_summaries(self.output, meta)

    def test_model_changed_tokens_rejected(self):
        meta = self.prepare()
        data = block_fixture(meta, 1)
        data["samples"][1]["token_ids"] = [3, 4]
        (self.output / "model" / "block1.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "token IDs"):
            model_summaries(self.output, meta)

    def test_profile_must_cover_linear_calls(self):
        meta = self.prepare()
        data = block_fixture(meta, 1)
        data["shape_profiles"].pop()
        (self.output / "model" / "block1.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "cover all linear"):
            model_summaries(self.output, meta)

    def test_outlier_is_retained(self):
        meta = self.prepare()
        data = block_fixture(meta, 1)
        next(r for r in data["samples"] if r["variant"] == "rb2x2")["wall_ms"] = 100
        (self.output / "model" / "block1.json").write_text(json.dumps(data))
        samples, pooled, *_ = model_summaries(self.output, meta)
        row = next(r for r in pooled if r["variant"] == "rb2x2")
        self.assertEqual(row["slow_samples_over_2x"], 1)
        self.assertEqual(row["max_ms"], 100)
        self.assertEqual(row["samples"], 6)
        self.assertEqual(len(samples), 18)

    def test_failed_resummary_removes_stale_success(self):
        meta = self.prepare(skip=True)
        summarize(self.output)
        write_csv(self.output / "micro.csv", micro_fixture(meta)[:-1])
        with self.assertRaises(ValueError):
            summarize(self.output)
        self.assertFalse((self.output / "summary.md").exists())

    def test_profile_wrappers_restore_and_classify(self):
        class Tensor:
            def __init__(self, shape): self.shape = shape
            def numel(self): return math.prod(self.shape)

        class Event:
            def record(self): pass
            def elapsed_time(self, end): return .25

        ops = SimpleNamespace(linear=lambda x, weight, bias=None, act=None: x)
        original = ops.linear

        class Model:
            c = SimpleNamespace(vocab=59264, text_hidden=2048)
            def encode_audio(self):
                return ops.linear(Tensor((1500, 1280)), Tensor((1280, 1280)))
            def next_token_probs(self):
                ops.linear(Tensor((64, 2048)), Tensor((2048, 2048)))
                return ops.linear(Tensor((1, 2048)), Tensor((59264, 2048)))

        model = Model()
        torch = SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kw: Event(), synchronize=lambda: None))
        with ShapeProfiler(model, ops, torch) as profiler:
            model.encode_audio()
            model.next_token_probs()
        self.assertIs(ops.linear, original)
        self.assertNotIn("encode_audio", vars(model))
        self.assertEqual({r["phase"] for r in profiler.rows()}, {"encoder", "decoder", "lm_head"})
        with self.assertRaises(RuntimeError):
            with ShapeProfiler(model, ops, torch):
                raise RuntimeError("fixture")
        self.assertIs(ops.linear, original)
        self.assertNotIn("next_token_probs", vars(model))

    def test_offline_orchestration_success_and_failure(self):
        for fail in (False, True):
            target = self.output / ("failure" if fail else "success")

            def fake_step(step, output):
                if fail:
                    raise RuntimeError("synthetic command failure")
                path = output / step.log
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Offline fixture, not a GPU result.\n")
                if step.log == "micro.log":
                    meta = json.loads((output / "metadata.json").read_text())
                    write_csv(output / "micro.csv", micro_fixture(meta))

            with mock.patch.object(runner.shutil, "which", return_value="/opt/cuda/bin/nvcc"), \
                    mock.patch.object(runner, "gpu_info", return_value=dict(name="offline fixture", sm="90")), \
                    mock.patch.object(runner.platform, "platform", return_value="offline"), \
                    mock.patch.object(runner.subprocess, "check_output", return_value="fixture\n"), \
                    mock.patch.object(runner, "run_step", side_effect=fake_step), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                before_env = {key: os.environ.get(key) for key in ("TMPDIR", "NVCC_CCBIN")}
                code = runner.main(["--output", str(target), "--skip-model", "--no-telemetry",
                                    "--cuda-tmpdir", str(self.output / "scratch"),
                                    "--variants", BASELINE, "rb2x2", "--repeats", "2"])
                self.assertEqual({key: os.environ.get(key) for key in before_env}, before_env)
            self.assertEqual(code, 1 if fail else 0)
            state = json.loads((target / "metadata.json").read_text())["status"]
            self.assertEqual(state, "failed" if fail else "complete")
            self.assertEqual((target / "summary.md").exists(), not fail)
            self.assertEqual((target / "FAILED.txt").exists(), fail)
            self.assertTrue((target / "source_sha256.json").exists())
            self.assertNotIn(b"\r", (target / "cases.csv").read_bytes())

    def test_nvcc_probe_failure_stops_before_task_builds(self):
        target, logs = self.output / "probe_failure", []

        def fake_step(step, output):
            logs.append(step.log)
            if step.log == "preflight/nvcc_probe.log":
                raise RuntimeError("synthetic nvcc host compiler preprocessor failure")

        with mock.patch.object(runner.shutil, "which", return_value="/opt/cuda/bin/nvcc"), \
                mock.patch.object(runner, "gpu_info", return_value=dict(name="offline fixture", sm="90")), \
                mock.patch.object(runner.platform, "platform", return_value="offline"), \
                mock.patch.object(runner.subprocess, "check_output", return_value="fixture\n"), \
                mock.patch.object(runner, "run_step", side_effect=fake_step), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = runner.main(["--output", str(target), "--skip-model", "--no-telemetry",
                                "--cuda-tmpdir", str(self.output / "scratch"), "--variants", BASELINE])
        self.assertEqual(code, 1)
        self.assertFalse(any(log.startswith("build/") for log in logs))
        self.assertFalse((target / "summary.md").exists())
        self.assertEqual(json.loads((target / "metadata.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
