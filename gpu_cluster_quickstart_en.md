# Edinburgh Teaching GPU Cluster Quick Start

## Connection overview

```text
Mac Terminal → SSH gateway → ICF head node → Slurm allocates a GPU → run on a compute node
```

> Logging in and getting a GPU allocation are separate steps. These instructions follow the lecture slides' `student.ssh.inf.ed.ac.uk` and `icf.inf.ed.ac.uk` route. If your course or the Computing team provides different hostnames, follow their latest instructions.

## 1. Check your network and account

- On campus, try connecting directly. Off campus, connect to the University VPN or Informatics VPN first.
- Make sure you have a DICE account and access to the course's Teaching cluster. Your DICE password is different from your University Login password.
- Replace `YOUR_UUN` in the commands below with your DICE username, for example `s1234567`.

## 2. SSH from your Mac to ICF

In Mac Terminal, run:

```bash
ssh YOUR_UUN@student.ssh.inf.ed.ac.uk
```

After logging in to the gateway, run:

```bash
ssh icf.inf.ed.ac.uk
```

You are now on an ICF head node. Head nodes are for preparing files and submitting jobs; it is normal for them not to have a GPU.

### Optional: Kerberos on macOS

Kerberos authenticates you and can reduce repeated password prompts. It does not remove the second SSH hop to ICF. After configuring `~/.ssh/config` according to the course or University instructions, run this on your Mac:

```bash
kinit -f YOUR_UUN@INF.ED.AC.UK
klist
ssh YOUR_UUN@student.ssh.inf.ed.ac.uk
```

Then, on the gateway, still run `ssh icf.inf.ed.ac.uk`.

## 3. Clone the course repository on the ICF head node

```bash
cd ~
git clone https://github.com/ed-aisys/edin-mls-26-autumn.git
cd edin-mls-26-autumn/cuda-tutorial
```

Clone into your ICF home directory so compute nodes can access the course code. The cluster home is not backed up, so keep another copy of important files on your own computer or GitHub.

## 4. Request an interactive GPU shell

Run this on the ICF head node:

```bash
sinfo -p Teaching

srun -p Teaching --gres=gpu:1 \
     --cpus-per-task=1 --mem=2G \
     --time=00:40:00 --pty bash
```

`sinfo` shows partition status; it does not reserve a GPU. `srun` queues for one GPU, one CPU, 2 GB of host RAM and up to 40 minutes. Once resources are allocated, it opens a Bash shell on a compute node. You may need to wait in the queue.

## 5. Check the GPU, then build and run an example

Once inside the GPU shell, run:

```bash
hostname
nvidia-smi -L
export PATH=/opt/cuda-12.8.0/bin:$PATH
nvcc --version
```

`hostname` shows the assigned compute node; `nvidia-smi -L` lists the GPU or MIG instance; `nvcc --version` checks the CUDA compiler.

Build and run the vector addition example:

```bash
cd ~/edin-mls-26-autumn/cuda-tutorial
make 2-vectoradd/memcpy
cd 2-vectoradd
./memcpy
```

The tutorial expects you to build and run on a GPU compute node. Do not run GPU calculations on the head node.

## 6. Exit and release the GPU

When you finish using the GPU shell, run:

```bash
exit
```

This ends the interactive Slurm job, releases the GPU and returns you to the ICF head node. Run `exit` again to leave ICF and return to your Mac's terminal.

## Common commands

| Command | Purpose |
|---|---|
| `sinfo -p Teaching` | Check Teaching partition status |
| `squeue -u "$USER"` | Check your queued and running jobs |
| `scancel JOB_ID` | Cancel a job |
| `srun ... --pty bash` | Request an interactive shell for debugging |
| `sbatch script.sh` | Submit a script job, useful for longer tasks |
