# Task 2: reductions

Files: `csrc/task2_reduce.cuh` (part A) and `csrc/task2_reduce.cu` (part B). Test: `make test2`. Model check: `./run.sh --cuda task1,task2`.

A reduction turns many numbers into one: a sum, a maximum, a mean. In task 1 every thread worked alone. Here threads have to combine their results, so they need ways to pass values to each other. In part A you build reduction helpers at two levels, the warp and the block. In part B you use them for RMSNorm and softmax. Task 4 uses your warp helpers again.

## Level 1: inside a warp

The GPU runs the threads of a block in groups of 32 called warps: threads 0 to 31 form warp 0, threads 32 to 63 warp 1, and so on. A thread's position inside its warp is its lane, `threadIdx.x % 32`. The 32 lanes of a warp execute the same instruction at the same time.

Lanes can read each other's registers directly, without going through memory:

```cuda
float other = __shfl_xor_sync(FULL_MASK, v, offset);
```

Every lane passes in its own `v` and gets back the `v` of lane `lane ^ offset` (`^` is bitwise XOR). `FULL_MASK` (`0xffffffff`) says that all 32 lanes take part, and all 32 must execute the call.

To add up the warp, every lane adds the value of its partner at offset 16, then 8, 4, 2 and 1. Here it is with 8 lanes and offsets 4, 2, 1:

```
lane:           0      1      2      3      4      5      6      7
start:          a      b      c      d      e      f      g      h
offset 4:      a+e    b+f    c+g    d+h    e+a    f+b    g+c    h+d
offset 2:     a+e+c+g  ...                         (lanes 0,2,4,6 now share one sum)
offset 1:     a+b+c+d+e+f+g+h  in every lane
```

After 5 steps all 32 lanes hold the total, without any shared memory or `__syncthreads()`.

## Level 2: across a block

A block of 256 threads has 8 warps. `block_reduce_sum` in `task2_reduce.cuh` is a complete worked example in four steps:

```
1. each warp reduces its 32 values                  8 warp sums (every lane holds its warp's sum)
2. lane 0 of warp w writes its sum to partial[w]    partial[] is __shared__
   __syncthreads()                                  wait until all 8 are written
3. every warp loads partial[lane] (0 for lane >= 8) and reduces again: every thread has the total
4. __syncthreads()                                  see below
```

Step 4 is easy to forget. The LayerNorm kernel calls `block_reduce_sum` twice. Without the barrier, a fast warp could finish the first call, start the second and write `partial[0]` while a slow warp is still reading `partial[0]` from the first call. That bug is a race: the result depends on timing, and the tests can pass anyway (see Try this).

## Online softmax: reducing two numbers at once

Softmax needs the maximum of a row and then the sum of `exp(x - max)`. Done naively, that is two separate passes over the data. There is a way to carry both at once. Summarise a set of numbers by two values:

```
m = max of the numbers
l = sum of exp(x - m) over the numbers
```

A single number `x` has the summary `{x, 1}`; the empty set has `{-inf, 0}`. Two summaries can be merged without seeing the numbers again, because `exp(x - m_new) = exp(x - m_old) * exp(m_old - m_new)`:

```
m = max(m1, m2)
l = l1 * exp(m1 - m) + l2 * exp(m2 - m)
```

For example, the numbers `1, 3, 2` have the summary `{3, e^-2 + e^0 + e^-1} = {3, 1.503}`. Split them as `{1, 3}` and `{2}`: the summaries are `{3, e^-2 + 1} = {3, 1.135}` and `{2, 1}`. Merging gives `m = 3` and `l = 1.135 * e^0 + 1 * e^(2-3) = 1.135 + 0.368 = 1.503`, the same.

This is the `MaxSum` struct and `maxsum_merge` (blank 2.4). It is also the core of FlashAttention, which you write in task 4.

## Part B: one block per row

RMSNorm, LayerNorm and softmax all work on a `[rows, cols]` matrix, one row at a time, and the launcher starts one block per row. The threads of the block walk along the row together:

```cuda
for (int c = threadIdx.x; c < cols; c += blockDim.x) { ... }
```

With 256 threads, thread 0 handles columns 0, 256, 512, ..., and thread 1 handles 1, 257, 513, .... In each round of the loop, the 256 threads read 256 neighbouring floats, which the GPU fetches as a few wide memory transactions (a coalesced access). The loop works for any number of columns, including fewer than 256.

Read the LayerNorm kernel first; it is complete and has exactly this shape. Your RMSNorm has the same shape with one reduction instead of two.

## The blanks

### Part A (`task2_reduce.cuh`)

2.1 `warp_reduce_sum`: one line in the loop.

2.2 `warp_reduce_max`: the same with `fmaxf`.

2.3 `block_reduce_max`: the four steps of `block_reduce_sum`, using `warp_reduce_max`.

<details><summary>Hint for 2.3</summary>

In step 3, lanes `>= num_warps` have no warp result to load. For a sum they load 0, which does not change the sum. For a max you need a value that can never be the maximum.

</details>

2.4 `maxsum_merge`: the merge rule above. If both `m` are `-INFINITY` (two empty summaries, or rows of masked scores in task 4), `exp(-inf - (-inf))` is `exp(nan)`; return `{-INFINITY, 0}` in that case.

### Part B (`task2_reduce.cu`)

2.5 `rmsnorm_kernel`: `y = x / sqrt(mean(x^2) + eps) * weight`. Follow the LayerNorm example: accumulate `x * x` over your columns, reduce, compute `rsqrtf(total / cols + eps)` (`rsqrtf(a)` is `1 / sqrt(a)` in one instruction), write.

2.6 softmax, first pass: each thread builds the `MaxSum` of its own columns, starting from `{-INFINITY, 0.0f}` and merging in `{x[c], 1.0f}` for each column. Then the block combines the threads' summaries: `row_max` is the block-wide max of the `m`s, and `row_sum` is the block-wide sum of each thread's `l * expf(m - row_max)`.

<details><summary>Hint for 2.6</summary>

The block-wide combine is two calls: `block_reduce_max` on `st.m`, then `block_reduce_sum` on the rescaled `st.l`. A thread with no columns (possible when `cols < blockDim.x`) still has `{-inf, 0}`; check that it contributes 0 to the sum and not `nan`.

</details>

2.7 softmax, second pass: `y = exp(x - row_max) / row_sum`. Multiplying by `1 / row_sum` is cheaper than dividing every element.

Every thread of the block must call `block_reduce_*`, because they contain `__syncthreads()`. Do not put them inside an `if` that only some threads enter.

## Test

```
make test2
```

When every case passes, the test ends with `Task 2 (reductions): ALL PASS`. As in task 1, each line also shows the time and the speed in GB/s.

The LayerNorm cases test your part A, because the given LayerNorm uses `warp_reduce_sum`. If only the LayerNorm cases fail, look at blank 2.1.

Common mistakes:

- Calling `block_reduce_*` from only some threads, which hangs the block or gives garbage.
- Loading 0 instead of `-INFINITY` for missing warps in `block_reduce_max`. All-negative rows then get a max of 0, and the "all negative" softmax case turns into `nan`.
- Using `blockDim.x` where you mean the number of warps, `blockDim.x / 32`.
- In softmax, rescaling `l` by `expf(row_max - m)` instead of `expf(m - row_max)`.

## Try this

1. Change `REDUCE_THREADS` to 32, 64, 128, 512 and 1024. Watch the "layernorm, audio encoder" and "softmax, vocabulary row" cases. Why do they react differently?
2. The vocabulary row is 59,264 floats, but its GB/s is far below that of the other large cases. How many blocks does the launch have, and how many of the GPU's SMs can that keep busy? Sketch (or write) a version that splits one long row over many blocks. `MaxSum` makes the second step easy.
3. Delete the final `__syncthreads()` (step 4) in `block_reduce_sum` and run `make test2`. Then run `compute-sanitizer --tool racecheck build/sm_XX/test_task2_reduce small` (`small` runs only the small cases, without timing, so that the slow sanitizer finishes in seconds). What do the test and the sanitizer each tell you? Put the barrier back afterwards.
4. In `rmsnorm_kernel`, replace the block reduction with `atomicAdd` on one `__shared__ float` (every thread adds its partial sum). Compare the time with your shuffle version.

## Questions for the report

1. Give an input row for which `block_reduce_max` returns a wrong maximum if missing warps load 0 instead of `-INFINITY`.
2. What is `expf(100.0f)` in float32? Explain how subtracting the row maximum keeps softmax correct for the "large inputs" case, and why the result is the same as without the subtraction.
3. How many times does your softmax read each element of `x` from global memory? How many times does the textbook version (max, then sum, then write) read it?
4. Describe, as a sequence of events between two warps, what can go wrong in LayerNorm without the final `__syncthreads()` in `block_reduce_sum`.
