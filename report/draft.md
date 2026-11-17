### 1 INTRODUCTION
Explain the system context and the technical question you investigate. Clarify why the question
matters, what baseline you start from, what you change, and the main result or trade-off you observe.

### ASR implementation and analysis
1. Q1.2
    The kernel computes `cos` and `sin` for every pair. The alternative is to compute a `[positions, half]` table of them once and read it in the kernel. Would the table make the kernel faster or slower? Count the bytes.
2. ?
3. Q3.2
4. Q4.2/Q4.4



### 2 SYSTEM AND TECHNICAL APPROACH
Describe only the supplied components needed to understand your investigation, then explain what
you completed, modified, or added. Make the technical question and design choices precise.
### 3 EXPERIMENTAL METHODOLOGY
Describe the data, baseline(s), metrics, hardware/software setup, experimental procedure, and con-
trols needed to understand and reproduce your comparison. Your evaluation should include an ap-
propriate quality/correctness measure and an appropriate systems-cost measure.
### 4 RESULTS AND ANALYSIS
Present evidence for a small number of precise claims. Analyse why the observed behaviour occurs,
including trade-offs, negative or unexpected results, and alternative explanations where relevant.
### 5 DISCUSSION AND CONCLUSION
Summarise what was learned, the most important quality–cost trade-offs, limitations, and what you
would change or investigate next.

