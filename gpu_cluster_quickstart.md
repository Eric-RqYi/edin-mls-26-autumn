# Edinburgh Teaching GPU 集群速查

## 连接路线

```text
Mac 终端 → SSH 跳板机 → SSH 到 ICF head node → Slurm 分配 GPU → compute node 上运行
```

> SSH 登录集群与申请 GPU 是两件事。下面按课件中的 `student.ssh.inf.ed.ac.uk` 和 `icf.inf.ed.ac.uk` 路线操作；如果课程或管理员给了不同主机名，以其最新说明为准。

## 1. 检查网络与账户

- 校园网内可直接尝试；校外先连接 University VPN 或 Informatics VPN。
- 准备好 DICE 账户和课程的 Teaching 集群权限。DICE 密码与 University Login 密码不同。
- 以下命令中的 `YOUR_UUN` 替换为自己的 DICE 用户名，例如 `s1234567`。

## 2. 从 Mac 登录 ICF

在 Mac 的 Terminal 运行：

```bash
ssh YOUR_UUN@student.ssh.inf.ed.ac.uk
```

登录跳板机后，再运行：

```bash
ssh icf.inf.ed.ac.uk
```

成功后当前 shell 位于 ICF head node。Head node 用于准备文件和提交作业；没有 GPU 是正常的。

### 可选：macOS Kerberos 登录

Kerberos 用于身份认证，避免每次 SSH 都输入密码；它不会省略跳转到 ICF 的第二条 SSH 命令。按课程/学校说明配置 `~/.ssh/config` 后，在 Mac 运行：

```bash
kinit -f YOUR_UUN@INF.ED.AC.UK
klist
ssh YOUR_UUN@student.ssh.inf.ed.ac.uk
```

然后仍需在跳板机上执行 `ssh icf.inf.ed.ac.uk`。

## 3. 在 ICF head node 克隆课程仓库

```bash
cd ~
git clone https://github.com/ed-aisys/edin-mls-26-autumn.git
cd edin-mls-26-autumn/cuda-tutorial
```

在 ICF 的 home 目录克隆，计算节点才能访问课程代码。重要文件还应备份到自己的电脑或 GitHub：集群 home 不保证备份。

## 4. 申请交互式 GPU shell

仍在 ICF head node 上运行：

```bash
sinfo -p Teaching

srun -p Teaching --gres=gpu:1 \
     --cpus-per-task=1 --mem=2G \
     --time=00:40:00 --pty bash
```

`sinfo` 只查看分区状态，不会预约 GPU。`srun` 会排队并申请 1 个 GPU、1 个 CPU、2 GB 主机内存，最长 40 分钟；资源分配成功后会打开 compute node 上的 Bash shell。排队期间可以等待。

## 5. 确认 GPU，编译并运行示例

进入 GPU shell 后运行：

```bash
hostname
nvidia-smi -L
export PATH=/opt/cuda-12.8.0/bin:$PATH
nvcc --version
```

`hostname` 显示分到的计算节点；`nvidia-smi -L` 显示 GPU/MIG 实例；`nvcc --version` 检查 CUDA 编译器。

编译并运行向量加法示例：

```bash
cd ~/edin-mls-26-autumn/cuda-tutorial
make 2-vectoradd/memcpy
cd 2-vectoradd
./memcpy
```

教程要求在 GPU compute node 上编译和运行，不要在 head node 上做 GPU 计算。

## 6. ASR / RAG 项目启动脚本

以下命令从仓库根目录运行，前提是已经进入 GPU 节点。模型、依赖和
RAG 课程数据应按各项目 README 在 head node 提前安装；启动脚本不下载文件。

```bash
# 在 head node 申请 GPU；现有脚本选择 H200 MIG 1g.18gb
bash start_cluster.sh

# 进入 GPU 节点后，选择一个项目
bash start_asr.sh
# 新 shell 的目录为 hw1-asr/，可执行：
bash run.sh --cuda none         # PyTorch 参考实现
make test3
bash run.sh --profile           # 当前完成的 CUDA kernels
exit                           # 返回原 GPU shell

bash start_rag.sh
# 新 shell 的目录为仓库根目录，默认 ASR_CUDA=none
python rag/scripts/run_audio_query.py --device cpu
exit                           # 返回原 GPU shell
exit                           # 释放 GPU allocation
```

也可以直接传入命令，运行结束后返回原 GPU shell：

```bash
bash start_asr.sh bash run.sh --cuda none
bash start_asr.sh make test3
bash start_rag.sh python rag/scripts/run_audio_query.py --device cpu
# kernels 验证通过后，RAG 可以切换为自定义 CUDA 实现
ASR_CUDA=all bash start_rag.sh python rag/scripts/run_audio_query.py --device cpu
```

两个脚本都应使用 `bash` 执行，不能 `source`。无参数时创建子 shell，环境配置
仅作用于该 shell；退出不会修改原 GPU shell。脚本配置环境并增量编译当前 GPU
的 ASR 库，启动本身不调用 ELM。需要检查 RAG 数据与 ELM 配置时，可手动运行
`python rag/scripts/check_setup.py`。

| 配置 | ASR | RAG |
|---|---|---|
| Python 环境 | `conda activate mls-asr` | 先激活 ASR 并保存解释器，再激活 `$HOME/mls-rag-env` |
| `PATH` | 将 `/opt/cuda-12.8.0/bin` 加入 `PATH` | 相同 |
| `TMPDIR` | 创建并使用 `$HOME/cuda-tmp`，避免之前的 nvcc 预处理失败 | 相同 |
| `NVCC_CCBIN` | `/usr/bin/g++`，显式指定 nvcc 的 host compiler | 相同 |
| `HF_HUB_OFFLINE` | `1`，使用 head node 下载的模型缓存 | 相同；不会阻止 ELM HTTP 请求 |
| `ASR_CUDA` | 默认 `all`，可设 `none` 或任务组 | 默认 `none`，验证 kernels 后可设 `all` |
| `RAG_ASR_PYTHON` | 无需配置 | 自动设为 ASR 环境的 Python 绝对路径 |
| `RAG_ASR_ROOT` | 无需配置 | 默认当前仓库的 `hw1-asr/` |
| ELM 配置 | 无需配置 | 在 `rag/.env` 填写 `ELM_API_KEY`，保留示例端点和模型设置；Python 自动读取 |

Conda 和 RAG 环境的非默认安装路径可通过 `CONDA_BASE` 和 `RAG_ENV_DIR`
覆盖。脚本保留现有 `HF_HOME` 等缓存设置；准备模型和运行时
应使用相同的缓存位置。`HF_TOKEN` 不是已缓存模型离线运行的必需项。
现有构建规则会处理 CUDA 测试的库路径，无需默认额外设置 `LD_LIBRARY_PATH`。
不要手动覆盖 Slurm 设置的 `CUDA_VISIBLE_DEVICES`，尤其是 MIG allocation。
RAG 示例的 `--device cpu` 只让查询编码器使用 CPU，ASR 仍在 GPU 上运行。
答案生成需要 GPU 节点能够访问 ELM 服务。

此前“排查 make 命令报错”的会话已验证：激活 `mls-asr` 后，设置
`TMPDIR="$HOME/cuda-tmp"` 与 `NVCC_CCBIN=/usr/bin/g++` 即可成功编译。
两个启动脚本均保留这组设置；当时未单独确定是哪一项导致原始报错。

## 7. 退出并释放资源

用完 GPU shell 后运行：

```bash
exit
```

这会结束交互式 Slurm 作业并释放 GPU，回到 ICF head node。再运行一次 `exit` 可退出 ICF，回到 Mac 的终端。

## 常用命令

| 命令 | 用途 |
|---|---|
| `sinfo -p Teaching` | 查看 Teaching 分区状态 |
| `squeue -u "$USER"` | 查看自己的排队/运行作业 |
| `scancel JOB_ID` | 取消指定作业 |
| `srun ... --pty bash` | 申请交互式 shell，适合调试 |
| `sbatch script.sh` | 提交脚本作业，适合较长任务 |

