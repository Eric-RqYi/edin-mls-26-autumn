#!/usr/bin/env python3
"""Build, validate and compare Task 3 M=1 variants on one allocated GPU node."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from summarize import CASE_NAMES, summarize

HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
FLAGS = "-O3 -std=c++17 -lineinfo -diag-suppress 177 -Xptxas=-v"


@dataclass
class Step:
    log: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are refused")
    parser.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Warps per M=1 block to compare (default: 1 2 4 8)")
    parser.add_argument("--sm", help="Optional architecture, e.g. 90; verified against visible GPU")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5, help="Microbenchmark warmup launches per sample")
    parser.add_argument("--iterations", type=int, default=20, help="Launches per CUDA-event timing sample")
    parser.add_argument("--repeats", type=int, default=7, help="Shuffled microbenchmark timing rounds")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--model-warmup", type=int, default=3)
    parser.add_argument("--model-runs", type=int, default=10)
    parser.add_argument("--skip-model", action="store_true", help="Skip all model loading/inference")
    parser.add_argument("--sanitizer", action="store_true", help="Also run memcheck and synccheck on small cases")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only; requires --sm, no GPU needed")
    args = parser.parse_args(argv)
    if (len(set(args.warps)) != len(args.warps)
            or any(not 1 <= value <= 32 for value in args.warps)):
        parser.error("--warps must be unique integers between 1 and 32")
    if args.sm is not None and not re.fullmatch(r"\d{2,3}", args.sm):
        parser.error("--sm must be numeric, for example 75, 86 or 90")
    if args.dry_run and args.sm is None:
        parser.error("--dry-run requires --sm; no GPU architecture is inferred offline")
    if min(args.jobs, args.iterations, args.repeats, args.model_runs) < 1:
        parser.error("jobs, iterations, repeats and model-runs must be positive")
    if min(args.warmup, args.model_warmup, args.seed) < 0:
        parser.error("warmup, model-warmup and seed must be non-negative")
    return args


def variants_for(output: Path, warps: list[int]) -> list[dict]:
    return [dict(name="tiled", enabled=0, warps=4, build=str(output / "build" / "tiled")),
            *[dict(name=f"m1_w{w}", enabled=1, warps=w,
                   build=str(output / "build" / f"m1_w{w}")) for w in warps]]


def make_steps(args: argparse.Namespace, output: Path, sm: str, variants: list[dict]) -> list[Step]:
    steps = []
    for variant in variants:
        name = variant["name"]
        flags = (f"{FLAGS} -DASR_TASK3_M1_ENABLED={variant['enabled']} "
                 f"-DASR_TASK3_M1_WARPS={variant['warps']}")
        make = ["make", f"SM={sm}", f"BUILD={variant['build']}", f"NVCCFLAGS={flags}"]
        steps += [Step(f"build/{name}.log", [*make, f"-j{args.jobs}", "all"]),
                  Step(f"tests/{name}.log", [*make, "test"])]

    executable = output / "build" / "benchmark_m1"
    steps.append(Step("build/benchmark_m1.log", [
        "nvcc", *shlex.split(FLAGS), f"-arch=sm_{sm}",
        f"-I{HW / 'csrc'}", f"-I{HW / 'tests'}", str(HERE / "benchmark_m1.cu"),
        "-o", str(executable), "-ldl",
    ]))
    libraries = [f"{v['name']}={Path(v['build']) / 'libasr_kernels.so'}" for v in variants]
    if args.sanitizer:
        for tool in ("memcheck", "synccheck"):
            steps.append(Step(f"sanitizer/{tool}.log", [
                "compute-sanitizer", "--error-exitcode", "99", "--tool", tool,
                str(executable), "--small", "--correctness-only", *libraries,
            ]))
    steps.append(Step("micro.log", [
        str(executable), "--csv", str(output / "micro.csv"),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
        "--repeats", str(args.repeats), "--seed", str(args.seed), *libraries,
    ]))
    if not args.skip_model:
        model_variants = [dict(name="torch"), *variants]
        random.Random(args.seed).shuffle(model_variants)
        for variant in model_variants:
            name = variant["name"]
            env = {"HF_HUB_OFFLINE": "1"}
            if name != "torch":
                env["ASR_LIB"] = str(Path(variant["build"]) / "libasr_kernels.so")
            steps.append(Step(f"model/{name}.log", [
                sys.executable, "-u", str(HW / "benchmark.py"),
                "--cuda", "none" if name == "torch" else "all",
                "--warmup", str(args.model_warmup), "--runs", str(args.model_runs),
                "--profile", "--tokens",
            ], env))
    return steps


def run_step(step: Step, output: Path) -> None:
    path = output / step.log
    path.parent.mkdir(parents=True, exist_ok=True)
    display = shlex.join([*[f"{k}={v}" for k, v in step.env.items()], *step.argv])
    print(f"\n>>> {step.log}: {display}", flush=True)
    env = dict(os.environ, ASR_DEBUG="0", CUDA_LAUNCH_BLOCKING="0", **step.env)
    with path.open("w") as log:
        log.write(f"Command: {display}\n")
        log.flush()
        with subprocess.Popen(step.argv, cwd=HW, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f"{step.log} failed with exit code {code}; see {path}")


def gpu_info() -> dict:
    # Use PyTorch's visible device, rather than nvidia-smi's first physical GPU.
    probe = """import json, torch
if not torch.cuda.is_available():
    raise SystemExit('No CUDA GPU visible; run inside a Slurm GPU allocation.')
p = torch.cuda.get_device_properties(0)
print(json.dumps(dict(name=p.name, sm=str(p.major)+str(p.minor),
    multiprocessors=p.multi_processor_count, memory_bytes=p.total_memory,
    torch_version=torch.__version__, torch_cuda=torch.version.cuda)))
"""
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"GPU preflight failed:\n{result.stderr or result.stdout}")
    return json.loads(result.stdout)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or HERE / "results" / f"{stamp}-{os.getpid()}").resolve()
    # The supplied Makefile does not quote BUILD paths in target declarations.
    if any(char.isspace() for char in str(output)):
        print("Output path must not contain whitespace (supplied Makefile limitation).", file=sys.stderr)
        return 1
    if any(source == output or source in output.parents
           for source in (HW / "csrc", HW / "tests", HW / "glm_asr_cuda")):
        print("Output must be outside the source directories being snapshotted.", file=sys.stderr)
        return 1
    if args.dry_run:
        for step in make_steps(args, output, args.sm, variants_for(output, args.warps)):
            print(f"[{step.log}] {shlex.join(step.argv)}")
            if step.env:
                print(f"  Environment overrides: {json.dumps(step.env, sort_keys=True)}")
        return 0

    created = False
    try:
        for program in ["make", "nvcc", "nvidia-smi", *(["compute-sanitizer"] if args.sanitizer else [])]:
            if shutil.which(program) is None:
                raise RuntimeError(f"Missing {program}; activate mls-asr and put CUDA on PATH")
        gpu = gpu_info()
        if args.sm and args.sm != gpu["sm"]:
            raise RuntimeError(f"--sm {args.sm} differs from visible GPU sm_{gpu['sm']}")
        variants = variants_for(output, args.warps)
        steps = make_steps(args, output, gpu["sm"], variants)
        output.mkdir(parents=True, exist_ok=False)
        created = True
        metadata = dict(
            started_utc=stamp, platform=platform.platform(), python=sys.version,
            gpu=gpu, variants=variants, flags=FLAGS, repeats=args.repeats,
            warmup=args.warmup, iterations=args.iterations, seed=args.seed,
            model_runs=args.model_runs, model_warmup=args.model_warmup,
            skip_model=args.skip_model, sanitizer=args.sanitizer,
            expected_micro_cases=CASE_NAMES,
            model_order=[Path(step.log).stem for step in steps if step.log.startswith("model/")],
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            tf32_override=os.environ.get("NVIDIA_TF32_OVERRIDE"),
            status="running",
        )
        try:
            metadata["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=HW, text=True, stderr=subprocess.DEVNULL).strip()
            metadata["git_branch"] = subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=HW, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            metadata["git_commit"] = "unavailable"
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # Preserve the actual inputs, including uncommitted kernel/script edits.
        shutil.copytree(HW / "csrc", output / "source" / "csrc")
        shutil.copytree(HW / "tests", output / "source" / "tests")
        for name in ("Makefile", "benchmark.py"):
            shutil.copy2(HW / name, output / "source" / name)
        shutil.copytree(HW / "glm_asr_cuda", output / "source" / "glm_asr_cuda",
                        ignore=shutil.ignore_patterns("__pycache__"))
        experiment_snapshot = output / "source" / "experiment"
        experiment_snapshot.mkdir()
        for source in HERE.iterdir():
            if source.is_file() and source.suffix in (".py", ".cu", ".md", ".sbatch"):
                shutil.copy2(source, experiment_snapshot / source.name)
        print(f"Results: {output}\nGPU: {gpu['name']} (sm_{gpu['sm']})", flush=True)
        run_step(Step("gpu.log", ["nvidia-smi", "-L"]), output)
        run_step(Step("nvcc.log", ["nvcc", "--version"]), output)
        for step in steps:
            run_step(step, output)
        report = summarize(output)
        metadata["status"] = "complete"
        metadata["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"\nExperiment completed. Summary: {report}", flush=True)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        message = f"Experiment failed: {error}"
        print(message, file=sys.stderr)
        if created:
            (output / "FAILED.txt").write_text(message + "\n")
            metadata_path = output / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                metadata["status"] = "failed"
                metadata["error"] = str(error)
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
