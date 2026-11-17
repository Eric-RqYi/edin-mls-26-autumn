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

## 6. 退出并释放资源

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

