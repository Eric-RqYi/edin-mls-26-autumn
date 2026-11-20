#!/usr/bin/env python3
"""Next-round GEMM builds, correctness checks, microbenchmarks and paired ASR blocks."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from results import summarize
from spec import BASELINE, CASE_FIELDS, CASES, FLAGS, VARIANTS, variant_flags

HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]


@dataclass
class Step:
    log: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New result directory; never overwrite")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--sm", help="Explicit architecture, required for offline dry-run")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cuda-tmpdir", type=Path,
                        help="Base for private nvcc temporary files (default: ASR_CUDA_TMPDIR or ~/cuda-tmp)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--model-warmup", type=int, default=3)
    parser.add_argument("--model-runs", type=int, default=10)
    parser.add_argument("--model-blocks", type=int, default=3, help="Independent model processes")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--sanitizer", action="store_true")
    parser.add_argument("--no-telemetry", action="store_true", help="Disable best-effort physical GPU sampling")
    parser.add_argument("--dry-run", action="store_true", help="Print commands; no GPU or output files needed")
    args = parser.parse_args(argv)
    if BASELINE not in args.variants or len(set(args.variants)) != len(args.variants):
        parser.error("--variants must contain tiled16 and have no duplicates")
    args.variants = [BASELINE, *[name for name in args.variants if name != BASELINE]]
    if args.sm is not None and not re.fullmatch(r"\d{2,3}", args.sm):
        parser.error("--sm must be numeric, e.g. 90")
    if args.dry_run and args.sm is None:
        parser.error("--dry-run requires --sm")
    if min(args.jobs, args.iterations, args.repeats, args.model_runs, args.model_blocks) < 1:
        parser.error("jobs/iterations/repeats/model-runs/model-blocks must be positive")
    if min(args.warmup, args.model_warmup, args.seed) < 0 or args.seed > 1000000000:
        parser.error("Warmups and seed must be nonnegative; seed must be <= 1e9")
    return args


def make_variants(output, names):
    return [dict(name=name, **VARIANTS[name], flags=variant_flags(name),
                 build=str(output / "build" / name),
                 library=str(output / "build" / name / "libasr_kernels.so")) for name in names]


def preflight_steps(output, sm, host_compiler):
    probe = output / "preflight"
    return [
        Step("preflight/host_compiler.log", [host_compiler, "--version"]),
        Step("preflight/host_preprocess.log", [
            host_compiler, "-std=c++17", "-E", "-x", "c++", str(probe / "host_probe.cpp"),
            "-o", str(probe / "host_probe.ii"),
        ]),
        Step("preflight/nvcc_probe.log", [
            "nvcc", *shlex.split(FLAGS), f"-arch=sm_{sm}", "--verbose", "--keep",
            "--keep-dir", str(probe / "keep"), "-c", str(probe / "cuda_probe.cu"),
            "-o", str(probe / "cuda_probe.o"),
        ]),
    ]


def make_steps(args, output, sm, variants, cuda_root):
    source = output / "source"
    experiment = source / "experiment"
    steps = preflight_steps(output, sm, os.environ.get("NVCC_CCBIN") or "/usr/bin/g++")
    for variant in variants:
        make = ["make", f"SM={sm}", f"BUILD={variant['build']}",
                f"SRC={source / 'csrc'}", f"NVCCFLAGS={variant['flags']}"]
        steps += [Step(f"build/{variant['name']}.log", [*make, f"-j{args.jobs}", "all"]),
                  Step(f"tests/{variant['name']}.log", [*make, "test"])]
    executable = output / "build" / "benchmark_gemm"
    libraries = [f"{v['name']}={v['library']}" for v in variants]
    steps.append(Step("build/benchmark_gemm.log", [
        "nvcc", *shlex.split(FLAGS), f"-arch=sm_{sm}",
        f"-I{source / 'csrc'}", f"-I{source / 'tests'}", str(experiment / "benchmark_gemm.cu"),
        "-o", str(executable), "-lcublas", "-ldl", "-Xlinker", "-rpath", "-Xlinker", str(cuda_root / "lib64"),
    ]))
    if args.sanitizer:
        for tool in ("memcheck", "racecheck", "synccheck"):
            steps.append(Step(f"sanitizer/{tool}.log", [
                "compute-sanitizer", "--error-exitcode", "99", "--tool", tool,
                str(executable), "--cases", str(output / "cases.csv"),
                "--small", "--correctness-only", *libraries,
            ]))
    steps.append(Step("micro.log", [
        str(executable), "--cases", str(output / "cases.csv"), "--csv", str(output / "micro.csv"),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
        "--repeats", str(args.repeats), "--seed", str(args.seed), *libraries,
    ]))
    if not args.skip_model:
        for block in range(1, args.model_blocks + 1):
            steps.append(Step(f"model/block{block}.log", [
                sys.executable, "-u", str(experiment / "model_runner.py"),
                "--manifest", str(output / "metadata.json"), "--block", str(block),
                "--output", str(output / "model" / f"block{block}.json"), "--hw-root", str(source),
            ]))
        # Also preserve the unmodified coursework benchmark's output for baseline.
        steps.append(Step("model/official_baseline.log", [
            sys.executable, "-u", str(source / "benchmark.py"), "--cuda", "all",
            "--warmup", "1", "--runs", "1", "--profile", "--tokens",
        ], dict(ASR_LIB=variants[0]["library"], HF_HUB_OFFLINE="1")))
    return steps


def run_step(step, output):
    path = output / step.log
    path.parent.mkdir(parents=True, exist_ok=True)
    display = shlex.join([*[f"{k}={v}" for k, v in step.env.items()], *step.argv])
    print(f"\n>>> {step.log}: {display}", flush=True)
    env = dict(os.environ, ASR_DEBUG="0", CUDA_LAUNCH_BLOCKING="0", **step.env)
    with path.open("w") as log:
        log.write(f"Command: {display}\n")
        log.write(f"TMPDIR: {env.get('TMPDIR', '<unset>')}\n")
        log.write(f"NVCC_CCBIN: {env.get('NVCC_CCBIN', '<unset>')}\n")
        log.flush()
        with subprocess.Popen(step.argv, cwd=output / "source", env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f"{step.log} failed with exit code {code}; see {path}")


def gpu_info():
    code = """import json, torch
if not torch.cuda.is_available():
    raise SystemExit('No CUDA GPU visible; run inside a Slurm GPU allocation.')
p = torch.cuda.get_device_properties(0)
print(json.dumps(dict(name=p.name, sm=str(p.major)+str(p.minor),
    multiprocessors=p.multi_processor_count, memory_bytes=p.total_memory,
    torch_version=torch.__version__, torch_cuda=torch.version.cuda)))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"GPU preflight failed:\n{result.stderr or result.stdout}")
    return json.loads(result.stdout)


def snapshot(output):
    source = output / "source"
    for name in ("csrc", "tests", "glm_asr_cuda"):
        shutil.copytree(HW / name, source / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("Makefile", "benchmark.py", "test_audio.wav", "test_audio.txt"):
        shutil.copy2(HW / name, source / name)
    (source / "experiment").mkdir()
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in (".cu", ".py", ".md", ".sbatch"):
            shutil.copy2(path, source / "experiment" / path.name)
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source.rglob("*")) if path.is_file()}
    (output / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


def prepare_build_environment(output, requested_tmpdir=None):
    """Match the working cluster setup; callers restore these two env variables."""
    inherited_tmpdir = os.environ.get("TMPDIR")
    base = requested_tmpdir or Path(os.environ.get("ASR_CUDA_TMPDIR") or Path.home() / "cuda-tmp")
    base = Path(base).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    # Never share nvcc scratch files across simultaneous jobs. Keep this directory
    # for diagnosis; do not delete caller-provided scratch roots or old run data.
    tmpdir = Path(tempfile.mkdtemp(prefix="task3-gemm-", dir=base))
    host_compiler = os.environ.get("NVCC_CCBIN") or "/usr/bin/g++"
    candidate = Path(host_compiler).expanduser()
    if candidate.is_dir():
        host_compiler = str(candidate / "g++")
    else:
        host_compiler = str(candidate)
    found = shutil.which(host_compiler)
    if found is None:
        raise RuntimeError(f"Host compiler is not executable: {host_compiler}; set NVCC_CCBIN to the compiler path")
    # Preserve the driver name (e.g. g++) rather than resolving a compiler symlink.
    host_compiler = str(Path(found).absolute())
    os.environ.update(TMPDIR=str(tmpdir), NVCC_CCBIN=host_compiler)
    probe = output / "preflight"
    (probe / "keep").mkdir(parents=True)
    (probe / "host_probe.cpp").write_text("#include <cstddef>\nint probe() { return sizeof(void*); }\n")
    (probe / "cuda_probe.cu").write_text(
        "#include <cuda_runtime.h>\n"
        "__global__ void probe(float* value) { if (threadIdx.x == 0) value[0] = 1.0f; }\n")
    return dict(tmpdir=str(tmpdir), tmpdir_base=str(base), inherited_tmpdir=inherited_tmpdir,
                host_compiler=host_compiler)


def telemetry_start(output):
    handle = (output / "telemetry.csv").open("w")
    command = ["nvidia-smi", "--query-gpu=timestamp,index,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw",
               "--format=csv", "--loop-ms=1000"]
    try:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
        return process, handle, command
    except OSError:
        handle.close()
        raise


def telemetry_stop(state, output):
    process, handle, command = state
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    handle.close()
    with (output / "telemetry.csv").open() as log:
        first = log.readline()
    return dict(command=command, status="captured" if first.startswith("timestamp") else "unavailable",
                exit_code=process.returncode, scope="physical GPUs; not per-MIG utilization")


def main(argv=None):
    args = parse_args(argv)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or HERE / "results" / f"{stamp}-{os.getpid()}").resolve()
    if any(char.isspace() for char in str(output)):
        print("Output path must not contain whitespace (coursework Makefile limitation).", file=sys.stderr)
        return 1
    if any(path == output or path in output.parents
           for path in (HW / "csrc", HW / "tests", HW / "glm_asr_cuda")):
        print("Output must be outside snapshotted source directories.", file=sys.stderr)
        return 1
    nvcc = shutil.which("nvcc")
    cuda_root = Path(nvcc).resolve().parent.parent if nvcc else Path("/opt/cuda-12.8.0")
    variants = make_variants(output, args.variants)
    if args.dry_run:
        tmp_base = args.cuda_tmpdir or os.environ.get("ASR_CUDA_TMPDIR") or Path.home() / "cuda-tmp"
        print(f"[build environment] private TMPDIR under {tmp_base}; no directory is created by dry-run")
        for step in make_steps(args, output, args.sm, variants, cuda_root):
            print(f"[{step.log}] {shlex.join(step.argv)}")
            if step.env:
                print(f"  Environment: {json.dumps(step.env, sort_keys=True)}")
        return 0

    created, telemetry, meta = False, None, None
    original_env = {key: os.environ.get(key) for key in ("TMPDIR", "NVCC_CCBIN")}
    try:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        for program in ["nvcc", "make", "nvidia-smi", *(["compute-sanitizer"] if args.sanitizer else [])]:
            if shutil.which(program) is None:
                raise RuntimeError(f"Missing {program}; activate mls-asr and put CUDA on PATH")
        gpu = gpu_info()
        if args.sm and args.sm != gpu["sm"]:
            raise ValueError(f"--sm {args.sm} differs from visible GPU sm_{gpu['sm']}")
        output.mkdir(parents=True, exist_ok=False)
        created = True
        meta = dict(status="running", started_utc=stamp, gpu=gpu, platform=platform.platform(),
                    python=sys.version, variants=variants, cases=CASES, repeats=args.repeats,
                    iterations=args.iterations, warmup=args.warmup, seed=args.seed,
                    model_blocks=args.model_blocks, model_runs=args.model_runs,
                    model_warmup=args.model_warmup, skip_model=args.skip_model,
                    sanitizer=args.sanitizer, m1_enabled=1, m1_warps=4,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    nvcc_ccbin=os.environ.get("NVCC_CCBIN"),
                    telemetry=dict(status="disabled" if args.no_telemetry else "pending"))
        for key, git_args in (("git_commit", ["rev-parse", "HEAD"]),
                              ("git_branch", ["branch", "--show-current"])):
            try:
                meta[key] = subprocess.check_output(["git", *git_args], cwd=HW,
                                                     text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                meta[key] = "unavailable"
        (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        snapshot(output)
        meta["build_environment"] = prepare_build_environment(output, args.cuda_tmpdir)
        meta["nvcc_ccbin"] = os.environ["NVCC_CCBIN"]
        (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        with (output / "cases.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(CASE_FIELDS)
            writer.writerows(CASES)
        print(f"Results: {output}\nGPU: {gpu['name']} (sm_{gpu['sm']})", flush=True)
        run_step(Step("gpu.log", ["nvidia-smi", "-L"]), output)
        run_step(Step("nvcc.log", ["nvcc", "--version"]), output)
        if not args.no_telemetry:
            try:
                telemetry = telemetry_start(output)
            except OSError as error:
                meta["telemetry"] = dict(status="unavailable", error=str(error))
        for step in make_steps(args, output, gpu["sm"], variants, cuda_root):
            run_step(step, output)
        if telemetry:
            meta["telemetry"] = telemetry_stop(telemetry, output)
            telemetry = None
        (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        report = summarize(output)
        meta.update(status="complete", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"\nExperiment complete. Summary: {report}", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ZeroDivisionError, KeyboardInterrupt) as error:
        message = f"Experiment failed: {error or 'interrupted'}"
        print(message, file=sys.stderr)
        if created:
            (output / "FAILED.txt").write_text(message + "\n")
            if meta:
                meta.update(status="failed", error=str(error))
        return 1
    finally:
        try:
            if telemetry:
                meta["telemetry"] = telemetry_stop(telemetry, output)
            if created and meta and meta["status"] == "failed":
                (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        finally:
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    sys.exit(main())
