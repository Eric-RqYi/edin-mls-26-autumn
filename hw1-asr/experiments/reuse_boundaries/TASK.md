# 任务 A：复用策略的适用边界与纯 warp 调整的 ASR 收益

请完成实验工具实现、必要验证、cluster 运行入口和结果分析。若已有可用且授权的 GPU 环境，运行并分析；若本机无 GPU，完成可执行代码、离线验证和明确的提交命令，说明尚未产生 GPU 结果。不要把本任务转成新的优化搜索。

## 报告目标与范围

报告只有六页，研究问题为：
1. 扩大每线程／线程块的计算范围以提高片上数据复用，能带来多大的内核和系统延迟收益？
2. 收益如何受到工作负载形状、资源需求和系统耗时分布的限制？

报告不讨论 direct softmax。所有新增报告实验固定 `ASR_ATTN_ONLINE_MODE=0`（standard）。保留旧代码和历史结果，不删除 direct 分支，不把旧 direct 结果重命名为 standard。

仅在 `hw1-asr/experiments/reuse_boundaries/` 增加工具、测试、README 和分析；复用现有 runner。不要更改生产默认、核心内核、课程测试、ops.py、model.py、benchmark.py、provided.cu 或共享 runner。另一窗口会开发 RAG 实验，避免写入它的目录。

## 先核实已有证据，避免重复运行

先读仓库 AGENTS.md（若存在）、experiment_log.md，以及：
- hw1-asr/experiments/task3_gemm/{README.md,spec.py,run_experiment.py,model_runner.py}
- hw1-asr/experiments/task4_attention/{README.md,attention_spec.py,run_attention.py}
- hw1-asr/experiments/task3_m1/README.md
- csrc/task3_gemm.cu、csrc/task4_attention.cu

已有记录（以 metadata、冻结源码和原始测量复核为准）：
- task3_gemm/results/20261117T192802Z-615737：6 配置×22 GEMM case，含 encoder、decoder、窄 N、卷积、projector，已有 rb4x4 直接形状对比。
- task4_attention/results/20261118T001456Z-748564：5 warp 数×2 softmax 的微基准。只抽取 standard 子集，可复用 14 个 attention case。
- task4_attention/results/20261118T082810Z-1093639：hybrid×attention factorial，但 attention 为 w16/direct，不能作为纯 warp 调整的系统结果。
- task3_m1/results/20261117T154855Z-483031：M=1 特化证据。

结果目录可能在其他 worktree 或 cluster 而没有同步到当前目录。先定位；若缺失，明确记录缺失，不从日志摘要伪造原始样本。旧 GEMM 的六种配置均固定 M=1 专用路径，M=1 行不能解释成未特化 rb4x4 与 tiled 的比较。

## 必做 A1：整理现有形状对比

从已有原始测量生成统一长表：case、M/N/K、bias/act、实际运行的 kernel 路径、variant、median ms、波动、相对 tiled16 speedup。按 encoder、decoder Q/O、gate/up、down、窄 N K/V、M=1 分组，保留卷积/projector 的完整表供附录或复现。

Attention 重新测量：形状、warp 数、耗时、配对 speedup、数值误差。正确性压力测试和真实模型代表形状分开标记，不能用压力测试数量代替模型覆盖。

## 必做 A2：定向补充边界扫描

不要穷举 M/N/K。固定如下小范围，测量前冻结清单和随机种子 21，不按结果追加冠军参数：

1. GEMM M 尾部扫描：N=2048、K=2048，M∈{16,31,32,33,48,59,63,64,65,71,80,96,128}。
2. 窄 N 扫描：M∈{32,64,65,96}，N∈{256,512,1024}，K=2048；另加 (64,512,1024)、(64,512,4096) 作为 K 邻近控制。
3. 大矩阵参考：复用或重新配对测量 encoder Q、FC1、FC2 的现有形状。
4. M=1：在 (1,512,2048)、(1,59264,2048) 上显式比较“关闭 M1 的 tiled16”“关闭 M1 的 rb4x4”“开启 M1/w4”。不得把三条路径混为一谈。

M>1 主对照：tiled16、tiled32、rb2x2、rb2x4、rb4x4、rb4x4+现有窄 N hybrid；全部固定 M1/w4、attention w4/standard。M1 对照独立构建。hybrid 规则继续是 M>1,N=512,K=2048，不据本轮结果修改。

Attention：复用已有 standard-only warp sweep，并补 seq_q=seq_k∈{32,128,512}，encoder-style 固定 batch=1、heads=kv_heads=20、D=64、noncausal，warp∈{1,2,4,8,16}。与原 encoder T=1500、原 decoder GQA/causal case 联合分析。补一个 seq_q=seq_k=65、batch=1、heads=16、kv_heads=4、D=128、causal case 检查尾部；已有单 query 作为低并行度边界。全程 standard，不加 direct。

这些是针对已知边界的验证，不应称为事先未看过结果的独立测试集。区分模型真实形状、诊断形状及配置选型用过的形状。

## 必做 A3：补测不含 direct 的 ASR 四组对照

现有入口可运行（从 hw1-asr 目录）：

```bash
sbatch --gres=gpu:h200_1g.18gb:1 experiments/task4_attention/run_cluster.sbatch \
  --stage factorial --selected-warps 16 --selected-online standard --sanitizer
```

四组均固定 M1/w4、rb4x4：base=w4/standard，hybrid=窄 N 分派+w4/standard，attention=w16/standard，combined=窄 N 分派+w16/standard。

显式传 standard，因为 runner 的 selected-online 默认是 direct。默认三个模型进程、每个十轮交错计时即可。检查逐 token 一致、wall time、linear/attention profile、形状与算子事件边界。报告纯 warp 收益、分派收益及组合，不沿用旧 7.72%/8.79%。

## 正确性、测量与机制证据

- 独立构建目录、冻结源码和 SHA256；同一 GPU/MIG 内配对。新换设备就重跑本轮配对对照，不能把旧 H200 毫秒数当新设备 baseline。
- 沿用已有 reference 和误差阈值、非默认 stream、尾部 guard；实现内核不得调用禁止的库。
- 补 hybrid 专项 memcheck/racecheck/synccheck：显式运行 M=2,N=512,K=2048，不能被旧 `--small` 工作量过滤掉；保存实际 case 与分派路径证据。覆盖通过后再执行长模型实验。
- 微基准默认 5 warmup、20 次调用取 CUDA-event batch mean、7 随机交错轮；数值参考、分配、H2D/D2H 不计入。
- 记录逻辑加载量/有效输出槽位、block/grid、ptxas registers、shared memory、spill；如有 Nsight Compute，仅对少量代表或退化 case 独立采样，保存版本及实际指标名，避免混入正式 wall time。无 profiler 时不要断言 memory-bound 或实测 occupancy 变化。
- 理论 global load 请求数不等于 DRAM 流量；同样，rb4x4 同时改变线程工作量和 block tile，不能把全部收益唯一归因于寄存器复用。
- 不删慢样本；按 case 配对统计，并明确随机轮次、独立进程和 GPU allocation 的层次。

## 交付与验收

交付 run_cluster.sbatch、runner、必要的离线验证、README、冻结配置清单、结果原始数据和 analysis.md。分析必须直接回答：
1. 现有真实形状中，rb4x4 哪些受益、哪些退化？
2. M=32/64 等 tile 边界附近有什么变化？统一扩大计算粒度的代价是什么？
3. 窄 N/M1 特化改善了哪些例外，是否对邻近形状同样有效？
4. Attention warp 收益在长度及 GQA/causal 条件下如何变化？
5. ASR 的纯 warp、hybrid 和组合收益各是多少？

生成最多两张正文候选图（GEMM 形状/边界；attention standard-only），以及一张四组 ASR 表；完整扫参表保留机器可读数据。不要为报告添加新优化。

向用户明确区分“工具验证完成”和“GPU 实验完成”。若无法访问已有原始结果或集群，只完成不依赖它们的工作，精确列出缺失路径/前置条件，不宣称产生了性能结果。
