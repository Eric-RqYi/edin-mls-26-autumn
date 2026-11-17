"""Offline tool tests with synthetic fixtures; these are not GPU measurements."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_experiment as runner
from summarize import CASE_NAMES, normalize, parse_model_log, summarize, summarize_micro


def micro_rows(variants=("tiled", "m1_w4"), repeats=2):
    rows = []
    for case in CASE_NAMES:
        for variant in variants:
            for repeat in range(1, repeats + 1):
                ms = 10.0 if variant == "tiled" else 5.0
                rows.append(dict(case=case, variant=variant, repeat=repeat, M=1,
                                 N=2, K=3, bias=0, act=0, ms=ms, weight_gbs=24 / (ms * 1e6),
                                 gflops=12 / (ms * 1e6), max_abs_error=1e-6, wrong=0))
    return rows


def write_micro(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_log(text="Concord returned to its place amidst the tents.", tokens=(13, 13)):
    return ("GPU: Offline fixture\n"
            f"  run 1: 6000.0 ms ({tokens[0]} tokens)\n"
            f"  run 2: 6200.0 ms ({tokens[1]} tokens)\n"
            "GPU time per operation, one transcription (5900.0 ms in kernels):\n"
            "  linear       cuda        123     5000.0  84.7%\n"
            "Peak GPU memory: 8.56 GiB\n"
            f"Transcription: {text}\n"
            "Expected: Concord returned to its place amidst the tents\n"
            "Accuracy: 100.0%\nStatus: PASS\n")


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_dry_run_has_no_side_effects(self):
        path = self.output / "not_created"
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = runner.main(["--dry-run", "--sm", "90", "--output", str(path)])
        self.assertEqual(code, 0)
        self.assertFalse(path.exists())
        self.assertIn("ASR_TASK3_M1_ENABLED=0", stdout.getvalue())
        self.assertIn("ASR_TASK3_M1_ENABLED=1", stdout.getvalue())
        self.assertIn("ASR_LIB", stdout.getvalue())

    def test_distinct_builds_and_original_test_targets(self):
        args = runner.parse_args(["--warps", "1", "4", "--skip-model"])
        variants = runner.variants_for(self.output, args.warps)
        steps = runner.make_steps(args, self.output, "90", variants)
        self.assertEqual(len({v["build"] for v in variants}), 3)
        self.assertEqual([s.argv[-1] for s in steps if s.log.startswith("tests/")], ["test"] * 3)
        self.assertFalse(any(s.log.startswith("model/") for s in steps))

    def test_invalid_warps_rejected(self):
        for values in (["0"], ["33"], ["4", "4"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(["--warps", *values])

    def test_dry_run_requires_explicit_architecture(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(["--dry-run"])

    def test_run_step_preserves_failure_log(self):
        step = runner.Step("failed/command.log", [runner.sys.executable, "-c",
                           "print('synthetic failure'); raise SystemExit(7)"])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            runner.run_step(step, self.output)
        self.assertIn("synthetic failure", (self.output / step.log).read_text())

    def test_micro_speedup_and_coverage(self):
        write_micro(self.output / "micro.csv", micro_rows())
        rows = summarize_micro(self.output / "micro.csv", ["tiled", "m1_w4"], 2)
        row = next(r for r in rows if r["case"] == "lm_head" and r["variant"] == "m1_w4")
        self.assertEqual(row["paired_speedup_median"], 2.0)
        self.assertEqual(row["median_ms"], 5.0)

    def test_missing_sample_rejected(self):
        write_micro(self.output / "micro.csv", micro_rows()[:-1])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize_micro(self.output / "micro.csv", ["tiled", "m1_w4"], 2)

    def test_duplicate_sample_rejected(self):
        rows = micro_rows()
        write_micro(self.output / "micro.csv", rows + rows[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_micro(self.output / "micro.csv", ["tiled", "m1_w4"], 2)

    def test_nonfinite_or_incorrect_micro_rejected(self):
        for field, value in (("ms", "nan"), ("ms", 0), ("wrong", 1)):
            rows = micro_rows()
            rows[0][field] = value
            write_micro(self.output / "micro.csv", rows)
            with self.assertRaises(ValueError):
                summarize_micro(self.output / "micro.csv", ["tiled", "m1_w4"], 2)

    def test_model_parser_uses_per_run_times(self):
        path = self.output / "model.log"
        path.write_text(model_log())
        row = parse_model_log(path, 2)
        self.assertEqual(row["median_ms"], 6100.0)
        self.assertEqual(row["tokens"], 13)
        self.assertEqual(row["linear_profile_ms"], 5000.0)
        self.assertEqual(normalize(row["transcript"]),
                         "concord returned to its place amidst the tents")

    def test_model_rejects_permissive_accuracy_false_positive(self):
        path = self.output / "model.log"
        path.write_text(model_log("Concord tents amidst place its to returned the extra."))
        with self.assertRaisesRegex(ValueError, "does not match"):
            parse_model_log(path, 2)

    def test_model_rejects_changing_token_count(self):
        path = self.output / "model.log"
        path.write_text(model_log(tokens=(13, 14)))
        with self.assertRaisesRegex(ValueError, "changing token"):
            parse_model_log(path, 2)

    def test_model_rejects_missing_profile_or_runs(self):
        path = self.output / "model.log"
        for text in (model_log().replace("linear", "other"),
                     model_log().replace("run 2:", "run 3:")):
            path.write_text(text)
            with self.assertRaises(ValueError):
                parse_model_log(path, 2)

    def test_skipped_model_and_stale_summary_invalidation(self):
        metadata = dict(variants=[dict(name=v) for v in ("tiled", "m1_w4")],
                        repeats=2, skip_model=True)
        (self.output / "metadata.json").write_text(json.dumps(metadata))
        write_micro(self.output / "micro.csv", micro_rows())
        path = summarize(self.output)
        self.assertIn("explicitly skipped", path.read_text())
        self.assertFalse((self.output / "model_summary.csv").exists())
        write_micro(self.output / "micro.csv", micro_rows()[:-1])
        with self.assertRaises(ValueError):
            summarize(self.output)
        self.assertFalse(path.exists())

    def test_full_summary(self):
        metadata = dict(variants=[dict(name=v) for v in ("tiled", "m1_w4")],
                        repeats=2, skip_model=False, model_runs=2)
        (self.output / "metadata.json").write_text(json.dumps(metadata))
        write_micro(self.output / "micro.csv", micro_rows())
        (self.output / "model").mkdir()
        for name in ("torch", "tiled", "m1_w4"):
            (self.output / "model" / f"{name}.log").write_text(model_log())
        summarize(self.output)
        with (self.output / "model_summary.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(float(r["speedup_vs_tiled"]) == 1.0 for r in rows))

    def test_failed_command_marks_run_failed(self):
        target = self.output / "failed_run"
        gpu = dict(name="offline fixture", sm="90")
        with mock.patch.object(runner.shutil, "which", return_value="offline-tool"), \
                mock.patch.object(runner, "gpu_info", return_value=gpu), \
                mock.patch.object(runner.platform, "platform", return_value="offline-platform"), \
                mock.patch.object(runner.subprocess, "check_output", return_value="fixture\n"), \
                mock.patch.object(runner, "run_step", side_effect=RuntimeError("synthetic failure")), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = runner.main(["--output", str(target), "--warps", "4", "--skip-model"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads((target / "metadata.json").read_text())["status"], "failed")
        self.assertTrue((target / "FAILED.txt").exists())
        self.assertFalse((target / "summary.md").exists())

    def test_offline_orchestration_publishes_complete_state(self):
        target = self.output / "synthetic_run"

        def fake_step(step, output):
            path = output / step.log
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Offline synthetic tool fixture, not a GPU result.\n")
            if step.log == "micro.log":
                write_micro(output / "micro.csv", micro_rows(repeats=2))

        with mock.patch.object(runner.shutil, "which", return_value="offline-tool"), \
                mock.patch.object(runner, "gpu_info", return_value=dict(name="offline fixture", sm="90")), \
                mock.patch.object(runner.platform, "platform", return_value="offline-platform"), \
                mock.patch.object(runner.subprocess, "check_output", return_value="fixture\n"), \
                mock.patch.object(runner, "run_step", side_effect=fake_step), \
                contextlib.redirect_stdout(io.StringIO()):
            code = runner.main(["--output", str(target), "--warps", "4", "--repeats", "2", "--skip-model"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((target / "metadata.json").read_text())["status"], "complete")
        self.assertTrue((target / "summary.md").exists())
        self.assertTrue((target / "source" / "csrc" / "task3_gemm.cu").exists())


if __name__ == "__main__":
    unittest.main()
