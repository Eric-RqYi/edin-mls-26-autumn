# 下一轮实验：M>1 GEMM 与按形状 profiling

上一轮 M=1 内核约有 4.58 倍局部加速，但只节省约 43 ms/转写。本轮固定 M=1 为 4 warp，研究占主要时间的其余 linear 层。默认实现仍为此前的 tiled16 + M=1/w4；实验不会自动把性能未知的候选设成生产默认。

## 1. 在 cluster 上运行

同步本地 `ASR-improvement` 的改动，按原 ASR README 准备 `mls-asr` 环境和模型 cache，然后在 cluster 的 `hw1-asr` 目录提交：

```bash
sbatch --gres=gpu:h200_1g.18gb:1 experiments/task3_gemm/run_cluster.sbatch --sanitizer
```

也可指定 `--gres=gpu:nvidia_rtx_a6000:1`。同一作业内的候选始终使用同一 GPU；不同 GPU/MIG 的原始时间不混为同一个对照。

脚本申请 2 CPU、16 GB CPU 内存、1 小时。可在 sbatch 命令中覆盖时限。默认 conda 位于 `~/miniconda3`，不同位置在提交前设置 `CONDA_BASE`；默认 `NVCC_CCBIN=/usr/bin/g++`，可用已有环境变量覆盖。计算节点使用本地模型 cache，不下载模型。

批处理和直接 Python 入口均使用此前集群已验证的 `~/cuda-tmp` 作为 nvcc 临时目录的根，每次运行在其下创建私有子目录，覆盖调度器继承的 `TMPDIR`。可用 `--cuda-tmpdir /absolute/path` 或 `ASR_CUDA_TMPDIR` 改变根目录（命令行优先）。运行信息记录原始/实际 TMPDIR 和 host compiler，退出 Python runner 时恢复调用前的环境变量。

先只验证内核、不加载模型：

```bash
sbatch --gres=gpu:h200_1g.18gb:1 experiments/task3_gemm/run_cluster.sbatch --skip-model --sanitizer
```

想先得到现有模型的真实形状表，可精简为 baseline 的诊断运行：

```bash
sbatch --gres=gpu:h200_1g.18gb:1 experiments/task3_gemm/run_cluster.sbatch \
  --variants tiled16 --model-blocks 1 --model-runs 1
```

这仍会执行 baseline 的测试和微基准，并对 baseline/PyTorch 做诊断 profile。只有一次模型计时，适合定位，不足以支持性能稳定性结论。

缩小正式对照范围：

```bash
sbatch --gres=gpu:h200_1g.18gb:1 experiments/task3_gemm/run_cluster.sbatch \
  --variants tiled16 rb2x4 rb4x4 --sanitizer
```

已有 GPU shell 中也可直接运行：

```bash
python -u experiments/task3_gemm/run_experiment.py --sanitizer
```

默认结果位于 `experiments/task3_gemm/results/UTC时间-PID/`，由本目录 `.gitignore` 忽略。`--output` 可指定新的绝对目录，拒绝覆盖已有目录；路径不能包含空白字符。完整 CLI 见 `--help`。

## 2. 固定因素和变化因素

| 版本 | M>1 实现 | block 线程数 | block 输出区域 | 每线程输出数 |
|---|---|---:|---|---:|
| tiled16（baseline） | 原 tiled，TILE=16 | 256 | 16×16 | 1 |
| tiled8 | 原 tiled，TILE=8 | 64 | 8×8 | 1 |
| tiled32 | 原 tiled，TILE=32 | 1024 | 32×32 | 1 |
| rb2x2 | register blocking | 256 | 32×32 | 4 |
| rb2x4 | register blocking | 256 | 32×64 | 8 |
| rb4x4 | register blocking | 256 | 64×64 | 16 |

固定项：所有版本 `M==1` 使用同一 4-warp 路径；其余 task 不变；FP32、不使用 fast math；bias 与原 exact GELU 保留；register-blocked K tile 固定为 16。每个候选独立编译，不复用其他候选的对象文件。

编译开关在 `csrc/task3_gemm.cu`，配置由 `spec.py` 统一生成：

- `ASR_TASK3_GEMM_IMPL=0/1`：tiled/register blocking。
- `ASR_TASK3_TILE=8/16/32`：原 tiled 内核的 tile 大小。
- `ASR_TASK3_RM`、`ASR_TASK3_RN`：register-blocked 内核每线程的输出行/列数。
- `ASR_TASK3_M1_ENABLED=1`、`ASR_TASK3_M1_WARPS=4`：本实验固定。

## 3. 理解 register blocking

线程 block 始终为 16×16。一个线程 `(ty, tx)` 计算：

```text
row = block_row + ty + i*16    i = 0 ... RM-1
col = block_col + tx + j*16    j = 0 ... RN-1
acc[i][j] += X[row, k] * W[col, k]
```

例如 rb2x4，每个线程持有 8 个累加器。对于同一个 k，读入 2 个 X 值和 4 个 W 值，再组成 8 次 FMA，而不为每个输出重复写出一套独立的 shared-memory 读取。它同时扩大了 block 的输出区域，因此也改变了 global-memory 复用和 grid 规模；不能把所有性能变化只归因于“寄存器更多”。

同一 block 的线程协作沿 K 连续加载 X 和 W；W 的 shared tile 继续有 `+1` padding。每轮 K tile 前后都有 block barrier。即使某线程对应的输出越界，它仍参与加载、同步，只在最后写回时检查边界。每线程输出增加还会增加寄存器需求，可能降低并发或引发 spill，这需要从 ptxas 与测量中判断。

保持这些问题可解释：哪个形状受益、为什么？M 较小和 M=1500 的最佳配置是否相同？局部变化乘以调用次数，是否解释了完整 ASR 的变化？

## 4. 每次运行的流程

1. 记录可见 GPU/MIG、编译器、软件版本、参数和 Git 状态标识，复制源码并生成 SHA256 清单。后续构建、模型调用和测试均使用冻结的源码快照。
2. 先记录 host compiler 版本，执行普通 C++ 预处理，再用 nvcc 编译一个最小 CUDA 文件（只编译，不做 GPU 计算），保存 verbose 日志及中间文件。通过后构建六个版本，各自运行未修改的完整课程 `make test`；任何失败立即停止。
3. 运行额外的 22 个 case。覆盖 M=1 控制组、奇数/尾部、K=1/7/31/33/129、bias/GELU、decoder M=59/64/71、N=512 的 K/V projection、encoder fc1/fc2、两次卷积和 projector。
4. 小工作量 case 对 CPU double reference，大 case 对 `CUBLAS_PEDANTIC_MATH` FP32 reference，误差沿用原测试 `atol=rtol=1e-3`。cuBLAS 仅用于实验参考，不进入任务实现。额外测试使用非默认 CUDA stream，并检查输出尾部 guard。
5. 指定 `--sanitizer` 时，对小 case 运行 memcheck、racecheck、synccheck。每种工具报告错误都使作业返回非零；这不代表大 case 也接受了 sanitizer。
6. 微基准每个样本预热 5 次、以 CUDA events 计时 20 次调用的均值，重复 7 轮，每轮随机打乱候选顺序。保存实际执行顺序。
7. 完整模型默认 3 个独立进程，每个进程加载一次权重。在每个进程中预热各版本，之后做 10 轮计时，每轮随机交错六个 CUDA 版本和 PyTorch reference。各版本使用同一份权重；通过原 `ops.py` 的 ctypes 接口切换库，保留全部库句柄。
8. 每次 warmup/计时/诊断生成均检查原始 token IDs 与 tiled16 reference 一致，转写与 expected 规范化后一致。每轮保存 token IDs、所选概率、wall time、CPU user/system 时间、上下文切换计数、UTC 时间和 allocated GPU memory 峰值。
9. 另做诊断生成，按 encoder/decoder/lm_head、`(M,N,K,bias,act)` 聚合 linear 事件时间与调用次数，并记录原算子级 profile。包装器仅在独立实验进程中生效，退出诊断上下文时恢复。检查所有 linear 调用均被覆盖。
10. 保留原 `benchmark.py` 对 baseline 的输出，供与课程工具核对；该次运行不混入交错样本。

课程原 `Makefile`、tests、`ops.py`、`model.py`、`benchmark.py` 和 `provided.cu` 均保持不变。核心修改只在 Task 3 文件内；默认不改 Task 4。

## 5. 结果与计时含义

| 文件 | 用途 |
|---|---|
| `metadata.json`、`gpu.log`、`nvcc.log` | 参数、设备、编译器和运行状态 |
| `preflight/*.log`、`preflight/keep/` | host compiler 版本、普通 C++ 预处理、最小 CUDA 编译的 verbose 诊断与中间文件 |
| `source/`、`source_sha256.json` | 实际执行/编译的源码快照与校验值 |
| `cases.csv` | 本次微基准的形状与 epilogue 清单 |
| `build/*.log`、`tests/*.log` | ptxas registers/spill/shared-memory 信息和完整课程测试 |
| `sanitizer/*.log` | 可选小 case 的三种 sanitizer 结果 |
| `micro.csv`、`micro_summary.csv` | 全部微基准样本、误差、GFLOP/s、配对 speedup 和波动 |
| `model/blockN.samples.jsonl` | 运行时逐条落盘的模型样本；中断后也可定位到已完成的轮次 |
| `model/blockN.json` | 完整独立进程的样本、reference、算子与形状 profile |
| `model_samples.csv` | 所有模型样本和诊断计数，保留慢样本并标记超过本 block 中位数两倍者 |
| `model_blocks.csv`、`model_summary.csv` | 每 block 和总体的 median/mean/IQR/stddev/max 与加速比 |
| `operation_profiles.csv` | 每个算子的诊断 profile 汇总 |
| `linear_shapes.csv` | 每个阶段、真实形状的调用次数、总时间和逐调用时间 |
| `model/official_baseline.log` | 未修改的课程 benchmark 输出 |
| `telemetry.csv` | 默认每秒采样物理 GPU 时钟、温度、功率；可用 `--no-telemetry` 关闭 |
| `summary.md` | 重点形状、完整模型以及 baseline 最昂贵形状的可读汇总 |
| `FAILED.txt` | 失败原因；失败时保留原始日志，不发布成功汇总 |

微基准只计 kernel launch 对应的 CUDA-event batch means，数据生成、参考计算、H2D/D2H、分配不计入。

上传在默认 stream 上进行，而额外测试使用非阻塞 stream，因此在参考计算/正确性检查前显式等待上传完成。这些同步位于计时区外；不能假定 pageable H2D 的同步 API 返回时 DMA 一定已完成。依据：[CUDA 12.8 API synchronization behavior](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html)。

模型 wall time 使用原 `ops.wall_clock_ms`，不含模型加载和音频预处理，包含生成中的 CPU/Python 开销；数据验证和落盘在计时区外。每轮与同 block、同轮 tiled16 的时间相除，再取每个 block 的配对加速比中位数；最终报告这些 block 中位数的中位数。三个进程仍在一次 allocation 中，不能当作三种独立硬件环境，也不能把 30 个相关样本包装成 30 次独立实验。

形状/算子 profile 是计时之外的诊断推理，GPU-event span 包括分配与 launch glue，包装器也有开销。不要直接把它当成无开销 kernel 时间；与微基准和正常 wall time 联合解释。CPU/context-switch 计数只是诊断线索，不能独自证明慢样本的原因。

telemetry 为 best-effort：`metadata.json` 记录 captured/unavailable。查询可能包含节点的多张物理 GPU，需要结合 `gpu.log` 的父 GPU UUID 对齐；它不是 per-MIG 利用率，也不会修改共享 GPU 时钟。UTC 时间可与模型样本对照。命令不可用/字段不支持时原错误仍留在日志中。

所有慢样本都进入 mean、stddev 和 max，不自动删除。不从一次结果自动选择新默认。检查不同 block 与形状是否一致，再决定后续分派策略。

重新汇总：

```bash
python experiments/task3_gemm/results.py experiments/task3_gemm/results/具体目录
```

只指定 `--skip-model` 时没有完整模型与 shape profile；汇总会明确标记这个限制。默认只测课程测试音频，逐 token 一致不构成数据集范围的 WER/CER 结论。实际 CUDA 编译、正确性与性能仍须在 cluster 验证。

### Host compiler 预处理失败

`20261117T171901Z-590775` 在首个版本构建 Task 1/2 时报告 `Failed to preprocess host compiler properties`，尚未到 Task 3 或数值测试。旧批处理脚本漏掉了启动脚本中已有的 TMPDIR 设置；旧日志未记录实际 TMPDIR，不能由这一条 fatal 信息唯一确定底层原因。

当前脚本已补齐已验证的 TMPDIR/host compiler 配置，并提前检查普通 C++ 与 CUDA 编译。同步修复后用原 sbatch 命令重新提交即可，生成新的结果目录；不要编辑旧失败目录的源码快照。若编译仍失败，`preflight/host_preprocess.log` 与 `preflight/nvcc_probe.log` 会保留更直接的错误、实际编译器命令和临时路径。参考：[nvcc 临时目录与 host compiler 环境变量](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html)。

## 6. 本地离线检查

从仓库根目录运行：

```bash
python3 -m unittest discover -s hw1-asr/experiments/task3_gemm -p 'test_experiment.py' -v
python3 -m unittest discover -s hw1-asr/experiments/task3_m1 -p 'test_tools.py' -v
bash -n hw1-asr/experiments/task3_gemm/run_cluster.sbatch
python3 hw1-asr/experiments/task3_gemm/run_experiment.py --dry-run --sm 90
```

两个 unittest 命令请分进程执行，避免两个实验目录中同名 runner 模块相互遮蔽。离线测试使用临时合成数据检查工具、覆盖验证、wrapper 恢复和失败处理，不产生可用于报告的 GPU 性能结果。dry-run 不编译、不运行命令，也不创建结果目录。
