# Task 4: attention with an online softmax

File: `csrc/task4_attention.cu`. Test: `make test4`. Model check: `./run.sh` (all four tasks).

This task puts the earlier ones together: dot products as in the GEMM, a tile of data shared by a block through shared memory, and the warp reductions and the `MaxSum` idea from task 2. The result works like FlashAttention: it never writes the attention scores to GPU memory.

## What attention computes

For one query vector `q` (one position, one head) and the key and value vectors `k_j`, `v_j` of the same head (`j = 0 .. seq_k-1`, each `head_dim` long):

```
s_j = (q . k_j) * scale              scale = 1 / sqrt(head_dim)
p_j = exp(s_j - max s) / sum_i exp(s_i - max s)          (softmax)
out = sum_j p_j * v_j                a vector of head_dim numbers
```

The output is an average of the value vectors, weighted by how well each key matches the query. Every position of every head does this: in one encoder layer that is 20 heads x 1500 queries, each against 1500 keys.

## Why not compute the scores first

The direct way is three kernels: compute all scores into a `[seq_q, seq_k]` matrix per head, softmax each row, multiply by V. For one encoder layer, that matrix is 20 x 1500 x 1500 floats, 180 MB, written once and read twice, although every score is needed only briefly. Its size grows with the square of the sequence length: at 15,000 positions it would be 18 GB per layer.

Instead, each query walks over the keys in tiles of 32 and keeps three running values:

```
m    the largest score seen so far
l    sum of exp(s_j - m) over the keys seen so far
acc  sum of exp(s_j - m) * v_j over the keys seen so far     (head_dim numbers)
```

`m` and `l` are the `MaxSum` of task 2; `acc` is new. When a tile raises the maximum from `m` to `m_new`, everything summed so far was computed relative to the old `m` and has to be multiplied by `corr = exp(m - m_new)`. After the last tile, `out = acc / l`.

A small example with 1-number values, four keys in two tiles of two:

```
scores s:  1   2  |  3   0          values v:  10  20  |  30  40

tile 1:  m = 2,  l = e^-1 + e^0 = 1.368,  acc = 10 e^-1 + 20 e^0 = 23.68
tile 2:  m_new = 3,  corr = e^(2-3) = 0.368
         p = [e^0, e^-3] = [1, 0.050]
         l   = 1.368 * 0.368 + (1 + 0.050)      = 1.553
         acc = 23.68 * 0.368 + (30 * 1 + 40 * 0.050) = 40.70
out = 40.70 / 1.553 = 26.21
```

Computing the softmax of all four scores directly gives the same 26.21.

## How the work is split

```
grid:   (ceil(seq_q / ATTN_WARPS), num_heads, batch)
block:  ATTN_WARPS = 4 warps = 128 threads, for 4 neighbouring queries of one head
warp w: query qi = blockIdx.x * 4 + w

for each tile of 32 keys (j0 = 0, 32, 64, ...):
    all 128 threads   copy keys and values j0 .. j0+31 into shared memory      (4.2)
    __syncthreads()
    each warp, for its own query:
        lane r        score s of key j0 + r                                     (4.3)
        the warp      online softmax update of m, l, acc                         (4.4)
        lane r        acc for output dims r, r+32, r+64, r+96                   (4.5)
    __syncthreads()   before the next tile overwrites shared memory
each warp writes acc / l                                                         (4.6)
```

The four warps of a block share each K/V tile, so each key and value is loaded from global memory once per four queries instead of once per query.

A lane has two jobs. While scoring, lane `r` owns key `j0 + r` of the tile and computes one full dot product with the query. While accumulating, lane `r` owns output dimensions `r`, `r + 32`, ... and adds up the contributions of all 32 keys. In between, the probabilities have to move between lanes: `p` of key `j0 + r` sits in lane `r`, and every lane needs all 32 of them. `__shfl_sync(FULL_MASK, p, r)` returns lane `r`'s `p` to every lane, a broadcast through registers.

### Layout and pointers

`q` and `out` are `[batch, seq_q, num_heads, head_dim]` and `k`, `v` are `[batch, seq_k, num_kv_heads, head_dim]`, as the linear layers produce them. The kernel sets up `k_head` and `v_head` for you: key `j` of the right head starts at `k_head + j * kv_stride`, with `kv_stride = num_kv_heads * head_dim`, the distance from one token to the next. Each warp's query is copied to shared memory (`qv`), already multiplied by `scale`.

### Shared memory

The kernel uses dynamic shared memory: one `extern __shared__ float smem[]` array whose size the launcher passes as the third launch parameter. It is cut into three parts:

```
k_tile  [32][head_dim + 1]      keys of the current tile
v_tile  [32][head_dim]          values of the current tile
q_all   [ATTN_WARPS][head_dim]  the queries of the block's warps
```

While scoring, lane `r` reads `k_tile[r * k_pitch + d]`: all lanes read the same `d` of different rows. The `+1` in `k_pitch = head_dim + 1` moves each row to a different bank, for the same reason as the `+1` in task 3.

### Grouped-query attention (GQA)

The decoder has 16 query heads and 4 key/value heads. Query heads 0 to 3 use key/value head 0, heads 4 to 7 use head 1, and so on. In the encoder both counts are 20, so every head uses its own.

### Causal masking

In the decoder a query may only look at keys at or before its own position. Query `qi` is at position `qi + (seq_k - seq_q)` of the key sequence; in this model `seq_q == seq_k`, but the test also has cases with fewer queries than keys, as in generation with a KV cache. The kernel computes `last_key` for each warp: keys after it get the score `-INFINITY`, which becomes `exp(-inf) = 0`.

The tile loop runs to `block_last_key`, the last key any query of the block needs, and not to each warp's own `last_key`. All threads of a block must run the same number of iterations, because the loop contains `__syncthreads()`. Tiles past the last key are skipped altogether, which halves the work in causal attention.

## The blanks

4.1 GQA: compute `kv_h` from `h`, `num_heads` and `num_kv_heads`.

4.2 Copy keys and values `j0 .. j0+31` into `k_tile` and `v_tile`, using all threads of the block, then wait. There are `32 * head_dim` elements of each; give thread `t` the elements `t, t + blockDim.x, ...` and turn each element number into a row `r` (key `j0 + r`) and a column `d`. Neighbouring threads then read neighbouring `d`, which is coalesced. Store 0 for keys at or past `seq_k`, so the last tile is never read out of bounds.

4.3 If key `j0 + lane` is visible to this warp's query, compute its score: the dot product of `qv` with row `lane` of `k_tile`. Otherwise leave `s = -INFINITY`.

4.4 The online softmax update. The comment in the code lists the steps. Use `warp_reduce_max` and `warp_reduce_sum` from task 2: every lane needs the maximum and the sum over the tile's 32 scores.

<details><summary>Hint for 4.4</summary>

If no lane has a visible key and nothing was seen before, `m_new` is `-INFINITY` and `exp(m - m_new)` is `exp(nan)`. Skip the update in that case and leave `p = 0`. The condition is the same in all 32 lanes (they all hold the same `m_new`), so calling `warp_reduce_sum` inside the `if` is safe. In this model that case only happens for the spare warps of the last block, whose `last_key` is -1.

</details>

4.5 Add the tile to `acc`: for each key `r = 0 .. 31` of the tile, get `p_r` with `__shfl_sync`, then `acc[c] += p_r * v_tile[r][lane + 32 c]` for each `c < dims`. All 32 lanes must call `__shfl_sync`.

4.6 Write `acc / l` to this warp's output row, if the warp has a real query.

### About the `acc` array

`acc` holds `dims = head_dim / 32` numbers per lane: 2 in the encoder, 4 in the decoder. Its size must be a compile-time constant, so it is declared with the maximum, 4. Loop over it as the given code does:

```cuda
#pragma unroll
for (int c = 0; c < MAX_DIMS_PER_LANE; ++c)
  if (c < dims) acc[c] *= corr;
```

With a constant loop count and `#pragma unroll`, the compiler turns the loop into four plain statements and keeps `acc` in registers. With `c < dims` as the loop condition, the compiler cannot tell which element `acc[c]` is at compile time and may put the array in slow local memory instead.

## Test

```
make test4
```

When every case passes, the test ends with `Task 4 (attention): ALL PASS`.

The CPU checks every query of the small cases and about 300 queries of the large ones. The "peaky scores" case has large queries, so the maximum often changes from one tile to the next; it fails if the `corr` rescaling is wrong.

Then run `./run.sh`. With all four tasks on your kernels, the transcription should pass.

Common mistakes:

- Only one `__syncthreads()` in the tile loop. The second one (after 4.5) is given; the one after the copy in 4.2 is yours.
- Comparing `j0 + lane` with `seq_k` instead of `last_key` in 4.3. Non-causal cases pass and causal ones fail.
- Forgetting to rescale `acc` by `corr`, or rescaling it after adding the new tile.
- Calling `__shfl_sync` or `warp_reduce_*` inside an `if` that only some lanes of the warp enter.

## Try this

1. Change `ATTN_WARPS` to 1, 2, 8 and 16 and time the audio encoder case. How does the number of times each key is loaded from global memory change, and how does the time follow?
2. Set `k_pitch = head_dim` (no `+1`) and time the encoder case again. Work out which banks the 32 lanes hit when they read `k_tile[lane * k_pitch + d]` with `head_dim = 64`.
3. Compare your attention with PyTorch's: `./run.sh --cuda task1,task2,task3 --profile` runs attention with PyTorch and everything else on your kernels. PyTorch does build the full score matrix, but it computes `Q K^T` and `P V` with cuBLAS. Compare the attention line of both profiles on your GPU, and think about which parts of your kernel do the most work per number loaded from shared memory.

## Questions for the report

1. How much memory would the full score matrix of one encoder layer take (20 heads, 1500 x 1500, float32)? How much shared memory does one block of your kernel use instead?
2. Why must all threads of a block run the same number of tile iterations? What could go wrong if each warp stopped at its own `last_key`?
3. What is `corr` when a tile does not raise the maximum? When can `m_new` be `-INFINITY`?
4. With `ATTN_WARPS = 4`, how many times is each key of one encoder head loaded from global memory per layer? What limits how large `ATTN_WARPS` can usefully be?
