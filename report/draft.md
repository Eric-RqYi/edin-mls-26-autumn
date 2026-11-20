### 1 INTRODUCTION
Explain the system context and the technical question you investigate. Clarify why the question
matters, what baseline you start from, what you change, and the main result or trade-off you observe.

1. Would more data reuse within wrap or thread be helpful? To what extend can it accelerate the task.
2. How is the improvement restricted by other factors?

The system for this project can be described as such: it first transcribes recorded speech - which for the scope of this project is a question - using automatic speech recognition (ASR); a retrieval-augmented generation (RAG) system applies the transcript to find relevant documents, which are used by a language model to answer the question from the recorded speech. The system completes all three steps in sequence.

We ask how much giving a GPU thread or block more work can reduce latency by reusing operands fetched from global memory, and whether extra resource use or reduced parallellism, in other words smaller or narrower input shapes, can limit that gain. As such the report investigates the trade-off between memory re-use and parallelism. Our baseline is the course ASR system with the four CUDA functionalities completed to specification. The modifications to the baseline made to investigate our research question do the following: we modify the matrix multiplication kernel so each thread computes more entries in a matrix product, as well as modify the attention kernel so each attention block serves more queries from the same key and value tiles. We also add specialised paths for a single-row matrix product and narrow-output matrices, where the general matrix multiplication kernel may leave much of its work idle, therefore causing inefficiency. On the reference recording, changing to the register-blocked general matrix kernel reduced model processing time by 43% from the baseline, with the single-row kernel and four attention warps held fixed. Across 30 development queries, the full set of changes made the core ASR 1.98x faster, and the complete audio-to-answer request 1.46x faster in paired measurements. The retrieved documents remained unchanged. These results suggest that grouping work helps most when the workload offers enough outputs to reuse data; short and narrow shapes can limit that benefit, exposing a cost of grouping more work together. Meanwhile answer generation through the external RAG service remains a substantial part of end-to-end time cost

### ASR implementation and analysis
1. ？
2. ?
3. Q3.1
4. Q4.4



### 2 SYSTEM AND TECHNICAL APPROACH
Describe only the supplied components needed to understand your investigation, then explain what
you completed, modified, or added. Make the technical question and design choices precise.
1. rb4*4 GEMM and attention warp
2. hybrid GEMM for other shapes

### 3 EXPERIMENTAL METHODOLOGY
Describe the data, baseline(s), metrics, hardware/software setup, experimental procedure, and con-
trols needed to understand and reproduce your comparison. Your evaluation should include an ap-
propriate quality/correctness measure and an appropriate systems-cost measure.

1. Increase data reuse by computing more output with the same thread or warp
    Rb 4*4 & attention warp performance compare with baseline
2. Check edge scenarios:
    The new kernels' performance on matrixes of different shapes
3. Improving the stand-out scenarios
    M=1 GEMM; narrow N GEMM performance vs. baseline
4. Testing system on wider samples
    performance check on RAG system

### 4 RESULTS AND ANALYSIS
Present evidence for a small number of precise claims. Analyse why the observed behaviour occurs,
including trade-offs, negative or unexpected results, and alternative explanations where relevant.
#### Improvements and limits from matrix data reuse
### 5 DISCUSSION AND CONCLUSION
Summarise what was learned, the most important quality–cost trade-offs, limitations, and what you
would change or investigate next.

## Modification checklist

section 1:  

    [] need more general research questions

    [] "Clarify why the question matters"

    [] Tradeoff (e.g., data reuse vs. computation waste?)

section 2:

    [x] Don't need descriptions on modules we didn't modify

    [x] Task 1-4 answers

    [] subsection 2.2

    [] flow chart

section 3:

    [] hyperlinks to table ？

    [x] Better list out clearly what experiments have we done, including the procedure.

    [x] Have to renounce baseline

    [] "your evaluation should include an appropriate systems-cost measure."

section 4:

    [] All of the numbers in the analysis should be visualised either in a table or a figure

    [] Names of a certain technical path must occurred first in the experiment methodology before being used in the result part

    [] Mainly reporting results, lacks analysis

section 5:

    [] should explicitly describe the trade-off
