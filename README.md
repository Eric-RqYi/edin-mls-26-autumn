# EDIN MLS 2026 Autumn

Welcome to the Machine Learning Systems course at the University of Edinburgh (Autumn 2026). This repository holds the course's code. The first part is the CUDA C++ examples from the lecture *GPU Architecture and CUDA Programming*. Every example is the code shown on the slides, as a complete program you can build and run on the teaching cluster.

The second part is Homework 1 (`hw1-asr/`): you write the CUDA kernels that a speech recognition model runs on. Start it after the CUDA tutorial; its [README](hw1-asr/README.md) has the instructions.

The RAG module (`rag/`) then connects ASR transcripts to document retrieval and
answer generation through the University ELM service. Its [guide](rag/README.md)
explains setup, course data installation, the baseline and evaluation. Use this
system to investigate a precise systems question through a controlled comparison
of quality and a relevant cost.

The shared [report resources](report/README.md) provide the LaTeX/Overleaf template
and PDF format checker for the coursework. The main body is limited to **6 pages**,
excluding references and optional appendices. Keep essential arguments and results
in the main body; reviewers are not obliged to read appendices.

## Where to start

| Part | Guide | What you do |
|---|---|---|
| CUDA tutorial | [cuda-tutorial/README.md](cuda-tutorial/README.md) | Build and run the lecture examples on the teaching cluster |
| ASR coursework | [hw1-asr/README.md](hw1-asr/README.md) | Complete and test the CUDA kernels used by speech recognition |
| RAG coursework | [rag/README.md](rag/README.md) | Run the audio-to-answer baseline and evaluate a systems investigation |
| Report | [report/README.md](report/README.md) | Explain your methodology, results, trade-offs and limitations |

Start with the CUDA tutorial, follow the ASR setup and task guide, then use the
RAG guide to run the complete system. ASR and RAG keep their code and tests in
separate directories. RAG calls the ASR model through a persistent worker in the
ASR Python environment; its own dependencies use a separate RAG environment.

## Quick Start: CUDA tutorial

These commands run the CUDA lecture examples, corresponding to slides 29–33 and
53. The allocation below is for the tutorial. Use the ASR guide's larger memory
allocation when running the ASR model or the complete RAG system.

```bash
# 1. Log in: two ssh hops (campus network or approved VPN)
ssh YOUR_UUN@ssh.inf.ed.ac.uk
ssh mlp.inf.ed.ac.uk

# 2. Clone the repository on the head node
git clone https://github.com/ed-aisys/edin-mls-26-autumn.git
cd edin-mls-26-autumn/cuda-tutorial

# 3. Ask Slurm for a GPU shell (the head node has no GPU)
srun -p Teaching --gres=gpu:1 --cpus-per-task=1 --mem=2G --time=00:40:00 --pty bash

# 4. On the compute node: put nvcc on PATH, build and run
export PATH=/opt/cuda-12.8.0/bin:$PATH
make
cd 1-first-kernel && ./hello
```

Leave the GPU shell with `exit` when you are done, so the GPU goes back to the pool.

## Repository Structure

```
edin-mls-26-autumn/
├── cuda-tutorial/                 # CUDA C++ examples from the lecture (slide numbers in each README)
│   ├── 0-environment/             # Teaching cluster: login and Slurm allocation (commands from the slides)
│   ├── 1-first-kernel/            # First kernel, function qualifiers
│   ├── 2-vectoradd/               # Example 0: vector add, host/device memory, student checkpoint
│   ├── 3-streams/                 # Example 1: CUDA streams and event dependencies
│   ├── 4-shared-memory/           # Example 2: matrix-vector multiply, explicit shared storage
│   ├── 5-tiled-gemm/              # Example 3: tiled GEMM, block barriers, event timing
│   ├── 6-optional-thrust-atomics/ # Optional: thrust containers, atomics
│   ├── common/check.h             # checkCudaErrors(), as used on the slides
│   └── Makefile                   # builds every example with nvcc -arch=native
├── hw1-asr/                       # Homework 1: speech recognition on your own CUDA kernels
│   ├── README.md                  # start here: setup, tasks, what to hand in
│   ├── docs/                      # a tour of the model and one guide per task
│   ├── csrc/                      # the kernels you complete (fill in the blanks)
│   ├── tests/                     # your kernels against a CPU reference, with timing
│   ├── glm_asr_cuda/              # the model in Python; calls your kernels
│   └── run.sh                     # build, then transcribe test_audio.wav
├── rag/                           # Retrieval-augmented answer generation
│   ├── README.md                  # Step-by-step setup and evaluation
│   ├── src/                       # ASR adapter and RAG pipeline components
│   ├── scripts/                   # Asset installation, query runs and evaluation
│   ├── tests/                     # Adapter and audio-interface tests
│   ├── data/rag_corpus/           # Course data and corpus embeddings
│   ├── models/enc_a/              # Local query encoder
│   └── extensions/                # Investigation-specific code
└── report/                        # Shared coursework report resources
    ├── latex_template/            # Report source and style files
    ├── report_template_overleaf.zip
    └── format_checker/            # PDF checks and six-page body limit
```

## Learning Path

| Phase | What to do | Slides | Goal |
|-------|------------|--------|------|
| **1. Setup** | `0-environment`: log in, get a GPU shell, check node, GPU and `nvcc` | 28–33 | A working GPU allocation |
| **2. First kernel** | `1-first-kernel` | 19, 36–37 | Grid, block, thread; host vs device code |
| **3. Example 0** | `2-vectoradd`, then the checkpoint `vector_add.cu` | 38–55 | Indexing, memory, launch configuration |
| **4. Example 1** | `3-streams` | 61–67 | Streams, events, overlapping copies |
| **5. Example 2** | `4-shared-memory` | 68–78 | Data reuse and `__shared__` |
| **6. Example 3** | `5-tiled-gemm` | 79–84, 87 | Tiling, barriers, timing against a baseline |
| **Optional** | `6-optional-thrust-atomics` | 57–60 | thrust, atomics |
| **Homework 1** | `hw1-asr`: RoPE, reductions, tiled GEMM, attention | | Run a speech recognition model on your own kernels |
| **RAG** | `rag`: install assets, inspect an audio query, evaluate retrieval and answers | | Connect ASR to retrieved evidence and answer generation |
| **Systems investigation** | Compare a technical change with the supplied baseline | | Explain quality and system-cost trade-offs |
| **Report** | `report`: template, argument and PDF checks | | Present reproducible evidence in a six-page main body |

## RAG data and ELM access

Download `rag-assets.zip` from the coursework materials and install it using the
[RAG guide](rag/README.md#step-2-install-the-course-data-once). The archive supplies
24,000 corpus records, 350 practice audio queries, 125 dev audio queries, public
relevance judgements, enc_a/enc_b corpus embeddings and the local enc_a query
encoder. Large data and model files are distributed separately from the Git source.

Apply for your own University ELM API key and configure `rag/.env` as described in
the RAG guide. Generation and answer evaluation make ELM requests. Start with the
guide's small trial before running a complete split, and keep the key local.

## Coursework experiments and report

Follow the ASR guide for kernel tasks, tests and task-specific experiments.
For the systems investigation, formulate a precise question, compare your change
with a documented baseline, and measure both quality and an appropriate cost.
Possible RAG directions include retrieval indexing, batching, reranking, context
selection, dynamic updates and representation migration. Explain why the observed
behaviour occurs, including trade-offs and negative results. Replacing the language
model alone is insufficient as a systems investigation.

Use the shared report template and preserve its format. The suggested section
headings are flexible: make the question, technical work, experimental controls,
results and limitations easy to find. Include the template's AI declaration and
group information, and run the PDF checker before submission. Follow the course
submission brief for hand-in arrangements and the module guides for technical
requirements.

## GPU Compatibility

The CUDA tutorial examples are plain CUDA C++ and need CUDA 11.6 or newer (for `-arch=native`). They were tested on the Teaching partition with CUDA 12.8 on NVIDIA GeForce RTX 2080 Ti (sm_75) and NVIDIA H200 MIG 1g.18gb (sm_90). Your allocation may give you a different GPU. Build on the compute node, so that `-arch=native` matches the GPU you were given.

## Resources

- **Lecture slides**: *GPU Architecture and CUDA Programming* (MLSys 2026, CUDA part). Each example README lists the slides it belongs to.
- **Teaching cluster**: [computing.help.inf.ed.ac.uk/teaching-cluster](https://computing.help.inf.ed.ac.uk/teaching-cluster), [Slurm](https://computing.help.inf.ed.ac.uk/slurm), [cluster tips](https://computing.help.inf.ed.ac.uk/cluster-tips)
- **CUDA**: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/), [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `nvidia-smi` or the program finds no GPU | You are on the head node. Run `srun ... --pty bash` first. |
| `nvcc: command not found` | `export PATH=/opt/cuda-12.8.0/bin:$PATH` on the compute node. |
| `nvcc fatal: Unsupported gpu architecture 'native'` / no GPU at compile time | Build on the compute node, or pass a target: `make ARCH=-arch=sm_75`. |
| `srun` waits for a long time | The partition is busy; `sinfo -p Teaching` shows its state but does not reserve a slot. |
| `cudaErrorUnsupportedPtxVersion` | The binary was built for another GPU or toolkit. Run `make clean && make` on the node you run on. |

## License

CC0 1.0 Universal. See [LICENSE](LICENSE).
