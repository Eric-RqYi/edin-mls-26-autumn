# 任务 B：多音频下的完整 RAG 延迟与输出一致性

请实现并验证可复现的多音频实验，提供 cluster 入口；有可用且授权的 GPU、数据和 ELM 配置时执行并分析，否则完成代码、离线检查和明确运行命令，列出实际缺失条件。当前任务是验证现有优化的系统价值，不增加新优化或 RAG 算法。

## 报告背景与修改范围

六页报告主线：扩大计算粒度提高片上复用 → GEMM register blocking / attention warp → 形状边界及 M1/窄 N 例外 → 完整 RAG 系统收益。

不讨论 direct softmax，所有测量固定 `ASR_ATTN_ONLINE_MODE=0`。旧 w16/direct 的 7.72% ASR 收益不是本次最终版本的测量。

只在 `rag/extensions/reuse_e2e/` 新增 runner、实验 worker/adapter、测试、README 和分析。复用现有代码，勿更改生产默认、课程任务测试、ops.py、model.py、benchmark.py、provided.cu、核心 kernel 或共享实验 runner。任务 A 在 hw1-asr/experiments/reuse_boundaries/ 工作，避免冲突。所需 CUDA 构建独立放入本任务结果目录。

先读 AGENTS.md（若存在）、experiment_log.md、rag/README.md，以及 rag/src/{asr_adapter.py,asr_worker.py,rag.py,generator.py,data.py,metrics.py}、rag/scripts/{run_audio_query.py,evaluate_rag.py} 和 hw1-asr/experiments/task3_gemm/model_runner.py。

## 必做 B1：冻结 baseline 和 final

统一 FP32、exact GELU、不启用 fast math，Task1/2/provided 相同，ASR_CUDA=all。两组都用自己的 CUDA 实现，不能以 PyTorch reference 替代 baseline：

| 因子 | original baseline | final（不含 direct） |
|---|---|---|
| ASR_TASK3_GEMM_IMPL | 0 | 1 |
| ASR_TASK3_TILE | 16 | 16 |
| ASR_TASK3_RM / RN | 2 / 2（未使用） | 4 / 4 |
| ASR_TASK3_M1_ENABLED | 0 | 1 |
| ASR_TASK3_M1_WARPS | 4（未使用） | 4 |
| ASR_TASK3_HYBRID_KV | 0 | 1 |
| ASR_ATTN_WARPS | 4 | 16 |
| ASR_ATTN_ONLINE_MODE | 0 | 0 |

显式写出所有开关：当前仓库默认已开启 M1，不能把不带编译参数的默认称为 original。此两组比较验证整体优化包；不要把总收益全部归因于某一个因素。单项归因由任务 A 和旧受控实验支持。

独立构建并测试两组库，保存完整 flags、源码快照、库 SHA256、GPU/MIG/软件信息。任务 A 的 hybrid 专项 sanitizer 结果可在哈希一致时复用；否则本任务显式对 M=2,N=512,K=2048 运行 memcheck/racecheck/synccheck，避免 `--small` 过滤该路径。

## 必做 B2：30 条多音频样本，预先冻结

从 dev split 的有效音频按 WAV 时长分成短/中/长三个数量近似相等的层，每层固定 seed=21 随机抽 10 条，共 30 条。分层只能使用预先可得的音频信息，不能按已测加速比筛选。保存 query ID、路径、时长、split、抽样代码/seed 和 manifest SHA256。

文件缺失或损坏在计时前记录，确定并冻结有效抽样框；少于所需样本时明确标成受限 subset，不从 practice 静默补齐，不在看过结果后替换慢样本。先用抽样外的两条 practice 音频做连通性检查与预热。

注意：音频时长不保证 encoder 的实际 tensor shape 改变。诊断记录实际 encoder/decoder 长度、生成 token 数；如模型统一 padding，应明确泛化变化来自哪些实际计算量，而不是仅依据音频秒数推断形状。

## 必做 B3：两层计时，保持真实完整执行

### 多音频 ASR 配对重复

对 30 条，每个版本每条做 3 次计时（共 180 次 ASR 推理），运行顺序按 query×round 配对随机化并记录实际顺序。固定生成配置与 max_new_tokens=100。每个版本预热 2 次，预热不计入结果。

记录音频加载/预处理、核心 model.generate（同步后 wall time）、ASR worker 往返总耗时、生成 token IDs/数量、transcript、EOS/截断状态、显存峰值；采集状态不应把逐次日志写盘纳入正式计时。形状/算子 profiling 在独立诊断推理中完成，不包进正式延迟；至少覆盖各长度层的代表音频。

无需为库切换重复加载权重。优先使用实验专用 worker：参考现有 model_runner，通过原 ops._library() 设置 ctypes 签名，保留所有句柄，在生成前设置 ops._lib；只在专用进程内操作并恢复，不编辑 ops.py。避免把两份约 9GB 模型同时常驻同一 18GB MIG。验证请求的 variant 与实际库哈希一致。

### 完整音频到答案计时

对同一 30 条和两版本，每条各做 3 次真实完整 pipeline（也是共 180 次 ASR + 180 次 ELM 生成；若能在同次执行中采集上述 ASR 指标，就共用这 180 次，勿额外重复 ASR 扫描）。同 query、同 round 内随机 AB/BA，query 顺序也随机；串行调用，不引入 batching/concurrency。每个版本总共 2 次完整预热即可（另最多 4 次 ELM），不要对每个 query 再做 ELM 预热。

主实验最多 184 次生成请求，不默认追加 LLM judge 或扩大样本。保存 API 尝试次数、重试、状态和 token usage（如可用）；失败单列，不偷偷重试直至获得好看的完整配对。若观察到服务端缓存，不伪造禁用缓存条件，不向 prompt 添加随机噪声，记录实际可观测状态。

固定 enc_a（CPU）、同一 corpus/embedding/id map、k=5、context builder、ELM model/temperature 和其余生成配置，不动检索算法。实验前用 setup 检查资产和配置；不输出 API key。

模型/encoder/index 加载放在请求计时外，单独记录 startup；保持 worker 和客户端预热状态。主 end_to_end 使用父进程单调时钟，从本次音频请求开始计到完整答案返回，包含音频读取/处理、ASR、IPC、query encoding、retrieval、context/prompt 构造、网络等待和生成，包含该次请求实际发生的重试。记录各阶段及未归属残差，避免阶段和总计边界不一致。不要用各阶段独立中位数相加当作实测总时间。

必须真正重新运行每版本 ASR，不共享 transcript cache、不回放 answer、不复用一次 ELM 延迟合成另一版本总耗时。可另外计算基于阶段占比的预测，但明确标记为模型估计，与实测分开。

## 正确性和下游质量

- 每条保存 baseline/final 的完整 token IDs 和 transcript，并核查每版本重复推理的稳定性。多音频不应套用课程单条音频的 EXPECTED 文本。
- 比较 query embedding、检索 scores/排名/IDs、context 和最终 prompt；对浮点向量给出明确容差，对 ID/字符串逐项核对。记录 token、transcript、top-k、prompt 各自一致率。
- 根据公开 qrels 计算当前 k 支持的检索指标，如 Recall@5、MRR@5/NDCG@5；若另算 top10/20，则标明诊断计算，与 k=5 生成上下文分开。
- 同 prompt 下 ELM 答案也可能随机变化，不要求答案文本一致，不把随机差异直接归因于 CUDA。
- 若 token/检索/prompt 有差异，不隐藏或直接宣称质量不变：保存差异样本、数值诊断及其下游影响，暂停发布“保持一致”的成功结论。没有人工参考转写就不计算 WER/CER。没有 judge 就不声称答案质量分数已验证；输入/证据一致性用于限定质量方面的结论。

## 统计与交付

以 query 为样本单位：先汇总每条的 3 次配对测量，再跨 30 query 统计；不要把 90 轮视为 90 条独立音频。报告 ASR 及端到端绝对毫秒节省、配对 speedup、mean/median/IQR，并按短/中/长层及实际 token 数解释差异。样本只有 30 条时避免强解读 p99。可按 query cluster bootstrap 给置信区间，固定 seed；保留全部慢样本和失败，不按收益删样本。

区分服务端/API 波动与本地 ASR 改善：若端到端区间包含零，如实报告“ASR 改善明确，完整系统收益尚不能从 API 噪声中可靠分辨”。不能为追求显著性无限追加请求。所有硬件比较在同型号/同 MIG 配置内完成；不把旧单音频比值外推成新数据集实测。

交付：runner、专用 worker/adapter、sbatch、离线测试、README、配置与样本 manifest、冻结源码/库哈希、原始 JSONL、per_query.csv、summary.csv、output_consistency.csv、retrieval_metrics、analysis.md。README 提供一个不调用 API 的 dry-run、一条连通性检查和一条正式提交命令。

正文只需一张阶段耗时图（baseline/final）和一张短/中/长或 query 配对结果图；机器可读数据保留全部样本。明确区分 warm inference 与 startup，ASR-only 与完整 RAG 实测。

不要根据缺失数据编造结果。无法访问 GPU/资产/ELM 配置时，完成不受影响的代码与验证，并列明缺失的具体前置条件及下一条运行命令。
