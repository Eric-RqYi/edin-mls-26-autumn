# Task 3: M=1 专用内核实验

研究问题：对 `lm_head` 的 `M=1, N=59264, K=2048`，warp 协作的矩阵向量乘能否比通用 tiled GEMM 更快，并且保持数值正确与转写一致？

本目录是额外的实验工具，不修改课程原始测试、`Makefile`、`ops.py`、`model.py`、`benchmark.py` 或 `provided.cu`。实际内核改动全部在 `csrc/task3_gemm.cu`。结果必须在 cluster GPU 上产生；本地 `--dry-run` 和 Python 测试不代表 CUDA 编译或 GPU 测试通过。

## 1. 内核与对照

- 默认在 `M==1` 时使用专用内核，每个 warp 合作计算一个输出，32 个 lane 沿 K 维连续读取权重并归约。不使用 shared memory 或 block barrier。
- 保留 FP32、可选 bias、原有 GELU 和调用方提供的 CUDA stream；`M>1` 继续使用原来的 `TILE=16` 实现。
- `ASR_TASK3_M1_ENABLED=0` 在编译时关闭分派，构建原 tiled baseline。`=1` 开启；默认开启。
- `ASR_TASK3_M1_WARPS` 控制每个 block 的 warp 数，默认 4。脚本默认扫描 1、2、4、8；允许 1 到 32。
- 每个版本使用全新的独立 `BUILD` 目录、相同源码与其余编译参数。没有 fast math、低精度转换或第三方计算库替代内核。
- warp 归约改变浮点累加顺序，因此要求误差满足原测试的 `atol=rtol=1e-3`，不要求与 tiled 逐 bit 相同。4 个 warp 是起始配置，不表示已测得最优。

## 2. 推荐：一次提交完整 Slurm 作业

先在 cluster head node 更新到本分支，并确保已按 ASR 原 README 建立 `mls-asr` 环境、下载模型。计算节点不需要联网。

从仓库根目录运行：

```bash
git switch ASR-improvement
cd hw1-asr
sbatch --gres=gpu:h200_1g.18gb:1 experiments/task3_m1/run_cluster.sbatch
```

也可以使用其他课程 GPU，例如 `--gres=gpu:nvidia_rtx_a6000:1`。所有版本在同一作业的同一 GPU 上顺序执行。脚本申请 2 CPU、16 GB CPU 内存、1 小时；可用 sbatch 参数调整时限。conda 默认位于 `~/miniconda3`，其他位置可在提交前设置 `CONDA_BASE`。

需要额外检查越界访问与 warp 同步时：

```bash
sbatch --gres=gpu:h200_1g.18gb:1 experiments/task3_m1/run_cluster.sbatch --sanitizer
```

查看 `task3-m1-JOBID.out`；结束时会打印结果目录与 `summary.md` 的路径。提交的是工作目录中的文件，请在运行期间保持源码不变。

## 3. 交互式 GPU 节点运行

在已有 GPU allocation 中激活环境：

```bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
conda activate mls-asr
cd hw1-asr
python -u experiments/task3_m1/run_experiment.py
```

尚未下载模型，或者只想先验证内核时：

```bash
python -u experiments/task3_m1/run_experiment.py --skip-model --sanitizer
```

精简为原版和默认 4-warp 版本：

```bash
python -u experiments/task3_m1/run_experiment.py --warps 4
```

更多选项：

```bash
python experiments/task3_m1/run_experiment.py --help
```

默认输出 `experiments/task3_m1/results/UTC时间-PID/`，已被 Git 忽略。可用 `--output /absolute/new/path` 指定位置；已有路径会被拒绝，不覆盖旧结果。路径不能包含空白字符，因为课程 Makefile 不支持这样的 BUILD 路径。单次完整运行需为多个构建目录和源码快照预留磁盘空间。

## 4. 每次完整运行做什么

1. 记录可见 GPU/MIG、SM 数、显存、PyTorch/CUDA 版本、编译器、Git commit/branch、配置和源码快照。SM 架构取自 PyTorch 实际可见的 GPU；手动 `--sm` 必须与它一致。
2. 构建 `tiled`、`m1_w1`、`m1_w2`、`m1_w4`、`m1_w8`。每个版本运行未改动的完整 `make test`，失败立即停止。
3. 运行额外 CPU double reference 校验，包括全部 59,264 个 lm_head 输出、K=1/7/31/32/33/129/2049/6144、非整齐 N、无 bias/bias、GELU 与 M=2/3 回退。输出尾部有 sentinel 检查，使用非默认 stream。
4. 如有 `--sanitizer`，对小 case 运行 memcheck 和 synccheck，工具报错会使作业失败。
5. 对所有版本使用相同输入做微基准：每个样本预热 5 次，CUDA events 计时 20 次调用的均值，重复 7 轮；每轮随机打乱版本顺序。可用 `--warmup`、`--iterations`、`--repeats`、`--seed` 调整。
6. 未指定 `--skip-model` 时，运行 PyTorch reference 与所有 CUDA 版本。每个版本在独立进程中调用未修改的 `benchmark.py`，预热 3 次、测量 10 次，并输出 profile 与最后一次计时推理的 token 文本/概率。用 `ASR_LIB` 指向对应库，无需覆盖默认库。模型使用本地 cache，`HF_HUB_OFFLINE=1`。
7. 检查最后一次计时转写是否与 expected 和 tiled 版本规范化后一致，所有计时 run 的 token 数是否一致。缺失/重复样本、数值错误、转写差异或外部命令失败均返回非零，不发布成功汇总。

参数修改后自动使用新的结果目录，避免 make 未追踪 NVCCFLAGS 变化造成旧对象复用。不要用默认 `./run.sh` 来选择实验库；原始脚本会构建默认目录。本实验直接调用原 `benchmark.py` 并设置 `ASR_LIB`。

## 5. 结果文件与解读

| 文件 | 内容 |
|---|---|
| `metadata.json`、`gpu.log`、`nvcc.log` | 配置、运行状态、GPU/MIG 与编译器信息 |
| `source/` | 本次内核、测试、模型桥接与实验工具的源码快照 |
| `build/*.log` | 构建记录及 ptxas 寄存器/shared memory 信息 |
| `tests/*.log` | 每个版本的完整课程测试输出，含 naive/cuBLAS 对照 |
| `sanitizer/*.log` | 可选的 memcheck/synccheck 结果 |
| `micro.csv` | 每个 case、版本、轮次的原始时间、执行顺序、误差与成本指标 |
| `micro_summary.csv` | 所有形状的 median/mean/stddev/min/max、配对 speedup 与误差 |
| `model/*.log` | 原 benchmark 的真实转写、每轮计时、内存与 profile |
| `model_summary.csv` | ASR 延迟分布、相对 tiled 加速比、token 数和 linear profile |
| `summary.md` | lm_head 与完整 ASR 的可读汇总；skip-model 时明确标记缺失 |
| `FAILED.txt` | 失败阶段与原因；失败后保留已生成的原始日志 |

优先比较 `lm_head` 的 median ms、波动、误差和加速比，再看完整 ASR 的收益。`fallback_m2/m3` 是 M>1 控制组，代码路径未变，时间应在测量波动内接近 baseline。

`weight_gbs = N*K*4 / time` 是权重逻辑字节数除以时间，不是硬件测得的 DRAM 流量；小 case 可能命中缓存。`min_io_bytes` 仅计一次输入/权重/输出/可选 bias，是算法最小逻辑流量，不是 tiled 或专用内核的实际事务计数。lm_head 权重为 485,490,688 bytes。

微基准 speedup 是每轮 `tiled_ms/variant_ms` 的中位数。模型 speedup 是不同进程的计时中位数之比，不是配对样本。模型计时不含权重加载和音频预处理，含生成过程中的 Python/CPU 开销；profile 是额外的一次推理，其 GPU 时间不可直接当作端到端延迟。

转写检查比原 benchmark 的词集合覆盖率更严格，但仅验证最后一次计时转写及每轮 token 数，不证明所有 token IDs 完全一致，也不是 WER/CER。默认只有课程测试音频；不能据此声称整个 ASR 数据集质量保持不变。性能更差也是有效实验结果，不自动修改默认 warp 配置或挑选更漂亮的数据。

重新汇总已有完整结果：

```bash
python experiments/task3_m1/summarize.py experiments/task3_m1/results/具体目录
```

## 6. 无 GPU 的本地检查

```bash
python experiments/task3_m1/run_experiment.py --dry-run --sm 90
python -m unittest discover -s experiments/task3_m1 -p 'test_tools.py' -v
bash -n experiments/task3_m1/run_cluster.sbatch
```

dry-run 只打印构建/运行命令，不执行命令、不创建结果目录。Python 测试使用合成数据验证工具逻辑，不产生可用于报告的 GPU 性能结果。
