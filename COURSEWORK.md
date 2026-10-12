# Coursework: ASR, RAG and the report

The coursework system takes an audio query and answers it using retrieved
documents. Start with the ASR module, then connect it to the RAG baseline.

| Part | Start here | What you work on |
|---|---|---|
| ASR and CUDA | [hw1-asr/README.md](hw1-asr/README.md) | Model setup, CUDA kernels and their tests |
| RAG | [rag/README.md](rag/README.md) | Retrieval, evidence prompts, answer generation and experiments |
| Report | [report/README.md](report/README.md) | Shared LaTeX template and PDF checker |

The modules live side by side. ASR and RAG use separate Python environments,
and the RAG adapter reuses the ASR model through a persistent worker. Download
`rag-assets.zip` from the coursework materials for the corpus, query audio,
embeddings and query encoder; the RAG guide explains installation.

Complete the ASR tasks according to its guide. For the systems investigation,
compare a technical change with a documented baseline, measure quality and the
relevant cost, and explain the result. Follow the course submission brief for
hand-in arrangements and the module guides for their technical requirements.

Use the shared report template. The main body is limited to **6 pages**,
excluding references and optional appendices. Reviewers are not obliged to
read appendices, so keep essential arguments and results in the main body.
