#!/usr/bin/env python3
"""Stage 1: attention sweep. Stage 2: paired GEMM x attention confirmation."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
GEMM_TOOLS = HW / "experiments" / "task3_gemm"
sys.path.insert(0, str(GEMM_TOOLS))
import run_experiment as runtime
import spec as gemm_spec
from attention_spec import ATTENTION_CASES, ATTENTION_FIELDS, FLAGS, WARPS, configs, variant_flags
from attention_results import summarize


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("sweep", "factorial"), default="sweep")
    p.add_argument("--selected-warps", type=int, choices=WARPS, default=4,
                   help="Attention warp count for factorial confirmation; no automatic winner selection")
    p.add_argument("--selected-online", choices=("standard", "direct"), default="direct",
                   help="Softmax update for factorial confirmation (default: direct)")
    p.add_argument("--variants", nargs="+", help="Subset, including the stage baseline")
    p.add_argument("--output", type=Path)
    p.add_argument("--sm")
    p.add_argument("--cuda-tmpdir", type=Path)
    for key, value in (("jobs", 2), ("warmup", 5), ("iterations", 20), ("repeats", 7),
                       ("seed", 21), ("model-warmup", 3), ("model-runs", 10), ("model-blocks", 3)):
        p.add_argument(f"--{key}", type=int, default=value)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--model", dest="model", action="store_true")
    mode.add_argument("--no-model", "--skip-model", dest="model", action="store_false")
    p.set_defaults(model=None)
    p.add_argument("--sanitizer", action="store_true")
    p.add_argument("--no-telemetry", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.selected_online = int(args.selected_online == "direct")
    available = configs(args.stage, args.selected_warps, args.selected_online)
    args.baseline = available[0]["name"]
    names = args.variants or [v["name"] for v in available]
    if len(set(names)) != len(names) or args.baseline not in names or set(names) - {v["name"] for v in available}:
        p.error(f"--variants must be unique valid names including {args.baseline}")
    args.variants = [args.baseline, *[n for n in names if n != args.baseline]]
    args.skip_model = not (args.model if args.model is not None else args.stage == "factorial")
    if args.sm is not None and not re.fullmatch(r"\d{2,3}", args.sm):
        p.error("--sm must be numeric, e.g. 90")
    if args.dry_run and args.sm is None:
        p.error("--dry-run requires --sm")
    if min(args.jobs, args.iterations, args.repeats, args.model_runs, args.model_blocks) < 1:
        p.error("Counts must be positive")
    if min(args.warmup, args.model_warmup, args.seed) < 0 or args.seed > 1000000000:
        p.error("Warmups/seed must be nonnegative; seed <= 1e9")
    return args


def gemm_cases(stage):
    if stage == "sweep":
        return [c for c in gemm_spec.CASES if c[0] in (
            "m1_lm_head_control", "decoder_q64", "decoder_kv64", "encoder_fc1")]
    return gemm_spec.CASES + [
        *[(f"kv_m{m}", m, 512, 2048, 0, 0) for m in (32, 48, 63, 65, 80, 96)],
        ("kv_n256_control", 64, 256, 2048, 0, 0),
        ("kv_n1024_control", 64, 1024, 2048, 0, 0),
        ("kv_k1024_control", 64, 512, 1024, 0, 0),
        ("kv_bias_gelu", 64, 512, 2048, 1, 1),
        ("m1_kv_control", 1, 512, 2048, 0, 0),
    ]


def make_variants(args, output):
    available = {v["name"]: v for v in configs(args.stage, args.selected_warps, args.selected_online)}
    return [dict(**available[name], flags=variant_flags(available[name]),
                 build=str(output / "build" / name),
                 library=str(output / "build" / name / "libasr_kernels.so")) for name in args.variants]


def make_steps(args, output, sm, variants, cuda_root):
    source, experiment = output / "source", output / "source" / "experiment"
    Step = runtime.Step
    steps = runtime.preflight_steps(output, sm, os.environ.get("NVCC_CCBIN") or "/usr/bin/g++")
    for v in variants:
        cmd = ["make", f"SM={sm}", f"BUILD={v['build']}", f"SRC={source / 'csrc'}", f"NVCCFLAGS={v['flags']}"]
        steps += [Step(f"build/{v['name']}.log", [*cmd, f"-j{args.jobs}", "all"]),
                  Step(f"tests/{v['name']}.log", [*cmd, "test"])]
    libraries = [f"{v['name']}={v['library']}" for v in variants]
    for name, src, cases, result in (
            ("attention", "benchmark_attention.cu", "attention_cases.csv", "attention_micro.csv"),
            ("gemm", "benchmark_gemm.cu", "cases.csv", "micro.csv")):
        exe = output / "build" / f"benchmark_{name}"
        steps.append(Step(f"build/benchmark_{name}.log", [
            "nvcc", *FLAGS.split(), f"-arch=sm_{sm}", f"-I{source / 'csrc'}", f"-I{source / 'tests'}",
            str(experiment / src), "-o", str(exe), "-lcublas", "-ldl",
            "-Xlinker", "-rpath", "-Xlinker", str(cuda_root / "lib64"),
        ]))
        if args.sanitizer:
            for tool in ("memcheck", "racecheck", "synccheck"):
                steps.append(Step(f"sanitizer/{name}_{tool}.log", [
                    "compute-sanitizer", "--error-exitcode", "99", "--tool", tool,
                    str(exe), "--cases", str(output / cases), "--small", "--correctness-only", *libraries,
                ]))
        steps.append(Step(f"{name}_micro.log", [
            str(exe), "--cases", str(output / cases), "--csv", str(output / result),
            "--warmup", str(args.warmup), "--iterations", str(args.iterations),
            "--repeats", str(args.repeats), "--seed", str(args.seed), *libraries,
        ]))
    if not args.skip_model:
        for block in range(1, args.model_blocks + 1):
            steps.append(Step(f"model/block{block}.log", [
                sys.executable, "-u", str(experiment / "model_runner.py"), "--hw-root", str(source),
                "--manifest", str(output / "metadata.json"), "--block", str(block),
                "--output", str(output / "model" / f"block{block}.json"),
            ]))
        steps.append(Step("model/official_baseline.log", [
            sys.executable, "-u", str(source / "benchmark.py"), "--cuda", "all",
            "--warmup", "1", "--runs", "1", "--profile", "--tokens",
        ], dict(ASR_LIB=variants[0]["library"], HF_HUB_OFFLINE="1")))
    return steps


def snapshot(output):
    runtime.snapshot(output)  # frozen coursework + tested shared runtime/model tools
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in (".py", ".cu", ".md", ".sbatch"):
            shutil.copy2(path, output / "source" / "experiment" / path.name)
    source = output / "source"
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.rglob("*")) if p.is_file()}
    (output / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def main(argv=None):
    args = parse_args(argv)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or HERE / "results" / f"{stamp}-{os.getpid()}").resolve()
    if any(c.isspace() for c in str(output)) or any(
            p == output or p in output.parents for p in (HW / "csrc", HW / "tests", HW / "glm_asr_cuda")):
        print("Output must have no whitespace and be outside source directories.", file=sys.stderr)
        return 1
    nvcc = shutil.which("nvcc")
    cuda_root = Path(nvcc).resolve().parent.parent if nvcc else Path("/opt/cuda-12.8.0")
    variants = make_variants(args, output)
    if args.dry_run:
        print(f"stage={args.stage}; baseline={args.baseline}; model={'skipped' if args.skip_model else 'enabled'}")
        for step in make_steps(args, output, args.sm, variants, cuda_root):
            print(f"[{step.log}] {runtime.shlex.join(step.argv)}")
        return 0
    created, meta, telemetry = False, None, None
    original_env = {k: os.environ.get(k) for k in ("TMPDIR", "NVCC_CCBIN")}
    try:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        for tool in ["nvcc", "make", "nvidia-smi", *(["compute-sanitizer"] if args.sanitizer else [])]:
            if shutil.which(tool) is None:
                raise RuntimeError(f"Missing {tool}; activate mls-asr and CUDA")
        gpu = runtime.gpu_info()
        if args.sm and args.sm != gpu["sm"]:
            raise ValueError("Requested SM differs from visible GPU")
        output.mkdir(parents=True)
        created = True
        meta = dict(status="running", started_utc=stamp, stage=args.stage, baseline=args.baseline,
                    selected_warps=args.selected_warps, selected_online=args.selected_online,
                    gpu=gpu, python=sys.version, platform=platform.platform(),
                    variants=variants, attention_cases=ATTENTION_CASES, cases=gemm_cases(args.stage),
                    repeats=args.repeats, warmup=args.warmup, iterations=args.iterations, seed=args.seed,
                    skip_model=args.skip_model, model_blocks=args.model_blocks, model_runs=args.model_runs,
                    model_warmup=args.model_warmup, sanitizer=args.sanitizer, profile_schema_version=2,
                    m1_warps=4, gemm_impl="rb4x4", cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    telemetry=dict(status="disabled" if args.no_telemetry else "pending"))
        for key, cmd in (("git_commit", ["rev-parse", "HEAD"]), ("git_branch", ["branch", "--show-current"])):
            try:
                meta[key] = runtime.subprocess.check_output(["git", *cmd], cwd=HW, text=True,
                                                            stderr=runtime.subprocess.DEVNULL).strip()
            except (OSError, runtime.subprocess.CalledProcessError):
                meta[key] = "unavailable"
        (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        snapshot(output)
        meta["build_environment"] = runtime.prepare_build_environment(output, args.cuda_tmpdir)
        meta["nvcc_ccbin"] = os.environ["NVCC_CCBIN"]
        (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        for name, fields, cases in (("cases.csv", gemm_spec.CASE_FIELDS, meta["cases"]),
                                    ("attention_cases.csv", ATTENTION_FIELDS, ATTENTION_CASES)):
            with (output / name).open("w", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(fields); writer.writerows(cases)
        print(f"Results: {output}\nGPU: {gpu['name']} (sm_{gpu['sm']})", flush=True)
        runtime.run_step(runtime.Step("gpu.log", ["nvidia-smi", "-L"]), output)
        runtime.run_step(runtime.Step("nvcc.log", ["nvcc", "--version"]), output)
        if not args.no_telemetry:
            try:
                telemetry = runtime.telemetry_start(output)
            except OSError as error:
                meta["telemetry"] = dict(status="unavailable", error=str(error))
        for step in make_steps(args, output, gpu["sm"], variants, cuda_root):
            runtime.run_step(step, output)
        if telemetry:
            meta["telemetry"] = runtime.telemetry_stop(telemetry, output); telemetry = None
        (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        report = summarize(output)
        meta.update(status="complete", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Complete: {report}", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ZeroDivisionError, KeyboardInterrupt) as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        if created:
            (output / "FAILED.txt").write_text(str(error) + "\n")
            if meta:
                meta.update(status="failed", error=str(error))
        return 1
    finally:
        try:
            if telemetry:
                meta["telemetry"] = runtime.telemetry_stop(telemetry, output)
            if created and meta and meta["status"] == "failed":
                (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        finally:
            for key, value in original_env.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value


if __name__ == "__main__":
    sys.exit(main())
