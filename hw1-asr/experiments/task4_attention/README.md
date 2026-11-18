# 下一轮：attention 消融与窄 N GEMM 分派

上一轮 H200 MIG 结果支持将 **rb4x4 + M=1/w4** 固定为实验 baseline。本轮先比较 attention 的 warp 数和 softmax 更新方式，再以完整 ASR 验证 attention 与 GEMM 分派的独立、组合收益。代码库的生产默认仍为 tiled16 + M=1/w4、attention w4/standard；各候选使用独立目录和编译参数。

## 在 cluster 上运行

同步本分支的全部修改，包括 `csrc/` 和 `experiments/task3_gemm/` 下的共享工具。只复制本目录无法运行。沿用已准备好的 `mls-asr` 环境和本地模型 cache，从 **cluster 的 `hw1-asr` 目录**提交。

第一步，扫描 10 个 attention 候选，先做数值检查和微基准：

```bash
sbatch --gres=gpu:h200_1g.18gb:1 experiments/task4_attention/run_cluster.sbatch \
  --stage sweep --sanitizer
```

默认不加载模型。`sweep` 覆盖 `warps=1/2/4/8/16 × standard/direct`，全部固定 rb4x4 和 M=1/w4，不开启 hybrid。`w4_standard` 为 baseline。查看新结果目录中的 `summary.md`、`attention_micro_summary.csv` 和 `build/*.log`，重点比较 encoder，同时检查 decoder/GQA、数值误差、寄存器与 spill。若希望这一阶段也比较完整 ASR 和真实 attention 阶段耗时，增加 `--model`；默认 3 个模型进程、每版本各 10 轮，耗时会增加。

第二步，选择 warp 数和 softmax 方式，再运行四组完整模型对照。以下以 **4 warp/direct 作为待验证配置**，不是已测冠军：

```bash
sbatch --gres=gpu:h200_1g.18gb:1 experiments/task4_attention/run_cluster.sbatch \
  --stage factorial --selected-warps 4 --selected-online direct --sanitizer
```

将 `--selected-warps` 替换为第一步支持的配置；若 standard 更合适，使用 `--selected-online standard`。选 4/standard 会让 attention 因子与 baseline 相同，适合重复对照，不能据此声称 attention 优化。`factorial` 默认执行模型实验，也可显式 `--no-model` 只做内核验证。

| 版本 | GEMM | Attention |
|---|---|---|
| base | rb4x4 | w4/standard |
| hybrid | rb4x4 + 窄 N tiled32 分派 | w4/standard |
| attention | rb4x4 | 选定 warp/softmax |
| combined | rb4x4 + 窄 N tiled32 分派 | 选定 warp/softmax |

所有版本的 M=1 路径均固定 w4。另交错测量 PyTorch reference。选项 `--variants base hybrid` 可先单独验证 GEMM 分派；sweep 的子集必须包含 `w4_standard`，factorial 的子集必须包含 `base`。只有四组齐全时才输出交互效应。

批处理默认 2 CPU、16 GB 内存、1 小时；可用 sbatch 参数调整时限和 GPU 类型。脚本按实际可见 GPU 自动选择 SM，结果记录硬件身份；不同 GPU/MIG 的原始毫秒数分开比较。在已有 GPU shell 中可将 sbatch 入口换成 `python -u experiments/task4_attention/run_attention.py` 并保留参数。

沿用上一轮修复过的 CUDA 12.8、`NVCC_CCBIN=/usr/bin/g++` 和 `~/cuda-tmp` 配置。可设置 `CONDA_BASE`、`NVCC_CCBIN`、`ASR_CUDA_TMPDIR`，或传 `--cuda-tmpdir /absolute/path`。Python runner 创建私有临时子目录，记录有效环境，并在退出时恢复调用前的环境变量。host C++ 预处理及最小 nvcc 编译必须通过后才构建候选；失败信息保存在 `preflight/`、`FAILED.txt`。

## 修改内容和数值检查

`csrc/task4_attention.cu` 新增两个编译开关：

- `ASR_ATTN_WARPS`：每个 block 协作的 query 数，扫描 1/2/4/8/16。
- `ASR_ATTN_ONLINE_MODE=0/1`：原 MaxSum 更新 / direct 更新。

direct 先算 `m_new=max(m,tile_max)`、`corr=exp(m-m_new)`，再以 `p=exp(s-m_new)` 更新 `l=l*corr+sum(p)` 与 `acc=acc*corr+pV`，减少原路径先算 `exp(s-tile_max)` 再算 `exp(s-m_new)` 的重复。全 masked tile 显式处理负无穷，完全 masked 的 query 输出定义为零。保留 K 的 shared-memory padding 和 block 内统一循环、barrier；无效 query 不提前退出。这个零输出约定扩展了 `seq_q>seq_k` 的 causal 边界测试，普通模型的 query 均有可见 key。

attention 微基准的 14 个 case 覆盖 encoder 1500×1500/D64、decoder/D128、batch>1、GQA、causal cache、单 query、D96、奇数 tile、uniform、全负分数、不断升高的峰值、peaky scores 和完全 masked 前缀。小 case 对 CPU double reference；大 encoder case 用 PEDANTIC FP32 cuBLAS 计算 QK/PV、CPU double 稳定 softmax，逐元素核对全部输出。cuBLAS 只用于测试参考。误差沿用课程 attention 测试 `atol=1e-4, rtol=1e-3`，不使用 fast math。

额外测试使用非默认 stream，上传后显式同步，检查尾部 guard。`--sanitizer` 对工作量不超过 1e6 的 case 运行 memcheck、racecheck、synccheck，每个候选均参与；这只覆盖小 case。所有候选还运行未修改的完整课程 `make test`。

`csrc/task3_gemm.cu` 新增 `ASR_TASK3_HYBRID_KV=0/1`：在专用 M=1 分支之后，仅将 **M>1、N=512、K=2048** 分派到 tiled32，其他形状维持候选原路径。规则只依赖矩阵维度，保留 bias/GELU 和调用者 stream。factorial 的 33 个 GEMM case 包括上一轮 22 个 case、M=32/48/63/65/80/96 的留出形状、相邻 N/K 控制组、bias+GELU 和 M=1 控制组。sweep 阶段只测 4 个 GEMM 控制 case。

## 计时与结果

每个结果目录默认写入 `experiments/task4_attention/results/UTC时间-PID/`。`--output` 可指定新目录，拒绝覆盖。源码、测试、共享 runner 都冻结到 `source/`，附 SHA256 清单；构建和模型使用冻结版本。旧实验结果及源码快照保持原样。

微基准每轮随机交错候选，默认 5 次预热、20 次调用取 CUDA-event batch mean、重复 7 轮。reference、数据传输、正确性比较和分配位于计时之外。完整模型默认 3 个独立进程，每个进程加载一次权重，在 10 轮中随机交错候选和 torch。每次生成检查精确 token IDs 与 baseline 一致、最终转写与 expected 一致，并保存概率、UTC 时间、CPU/上下文切换和显存峰值。单条课程音频不构成数据集 WER/CER 评价。

诊断推理与正常模型计时分开。新的 ShapeProfiler 直接关联原 `ops._timed` 创建的事件，不再额外创建外层 CUDA events；linear、attention 的形状时间与各自原算子总时间必须在容差内一致，调用数必须相同。attention 单独按 encoder/decoder、batch、heads/KV heads、seq_q/seq_k、head_dim、causal 汇总。原事件跨度仍包含原算子内的分配及 launch glue，应结合微基准解释。

| 文件 | 用途 |
|---|---|
| `metadata.json`、`gpu.log`、`nvcc.log` | 配置、实际硬件、编译环境与状态 |
| `source/`、`source_sha256.json` | 本次实际执行的冻结源码 |
| `build/*.log`、`tests/*.log`、`sanitizer/*.log` | 编译资源、原课程测试、小 case sanitizer |
| `attention_micro.csv`、`attention_micro_summary.csv` | 所有 attention 样本、误差、配对 speedup |
| `micro.csv`、`micro_summary.csv` | GEMM 回归和 hybrid 留出形状验证 |
| `model/blockN.json`、`model/blockN.samples.jsonl` | 独立进程模型数据；JSONL 逐次落盘 |
| `model_summary.csv`、`model_blocks.csv`、`model_samples.csv` | 模型统计、配对 speedup、全部原始慢样本 |
| `operation_profiles.csv`、`linear_shapes.csv`、`attention_shapes.csv` | 真实算子、阶段和形状耗时 |
| `profile_boundary_checks.csv` | linear/attention 形状总和与原算子时间核对 |
| `factorial_effects.csv` | 同 block、同轮的两种单项节省、组合节省和交互项 |
| `telemetry.csv` | best-effort 物理 GPU 采样；不是 per-MIG 利用率 |
| `summary.md`、`FAILED.txt` | 校验后的报告 / 失败原因 |

加速比按同 block、同轮与阶段 baseline 配对，再报告 block 中位数的中位数。交互项为 `combined - hybrid - attention + base`，正值表示实际组合比简单相加预测更慢。保留全部慢样本，分别标记超过本 block 中位数 5% 与 2 倍者，mean/std/max 均包含它们。三个模型进程仍位于同一 allocation，不当作三种独立硬件条件；不会自动选择或部署冠军。

重新校验并汇总已有新实验目录：

```bash
python experiments/task4_attention/attention_results.py experiments/task4_attention/results/具体目录
```

任何缺样本、错误输出、token 不符或事件边界不符都会返回非零，不发布成功报告。`--no-model` 的报告明确只支持微基准结论。

## 本地离线检查

从仓库根目录执行，各套 unittest 分进程运行：

```bash
python3 -m unittest discover -s hw1-asr/experiments/task4_attention -p test_attention.py -v
python3 -m unittest discover -s hw1-asr/experiments/task3_gemm -p test_experiment.py -v
python3 -m unittest discover -s hw1-asr/experiments/task3_m1 -p test_tools.py -v
bash -n hw1-asr/experiments/task4_attention/run_cluster.sbatch
python3 hw1-asr/experiments/task4_attention/run_attention.py --dry-run --sm 90 --stage sweep --sanitizer
python3 hw1-asr/experiments/task4_attention/run_attention.py --dry-run --sm 90 --stage factorial --selected-warps 4
```

离线测试使用临时合成数据验证配置、覆盖检查、配对统计、事件复用、失败处理、快照和环境恢复，不产生 GPU 实验结论。dry-run 不创建目录、不运行命令。当前本机无 nvcc/NVIDIA GPU，实际 CUDA 编译、数值正确性和性能均需在 cluster 上验证。
