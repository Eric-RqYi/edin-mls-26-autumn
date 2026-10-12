# Task 4: attention

File: `csrc/task4_attention.cu`. Build: `make`. Test: `make test4`. Model check: `./run.sh` (all four tasks).

This is the last and largest task, and it reuses what you built before: dot products as in the GEMM, a tile that a block shares through shared memory, and the warp reductions and `MaxSum` from task 2. Read the sections in order; each one adds one idea, and section 7 lists the blanks.

## 1. What attention computes

In attention, every position of the sequence collects information from the other positions. The linear layers give each position three vectors of `head_dim` numbers:

- the query `q`: what this position is looking for
- the key `k`: what this position offers to the others
- the value `v`: what it passes on when it is chosen

For one query and the keys and values `j = 0 .. seq_k-1` of the same head:

```
+----------------------+     +----------------------+     +----------------------+
| scores               |     | softmax              |     | weighted average     |
| s_j = (q . k_j)      | --> | p_j = exp(s_j) / sum | --> | out = sum of p_j v_j |
|       * scale        |     | (the p_j add to 1)   |     |                      |
+----------------------+     +----------------------+     +----------------------+
```

`q . k_j` is the dot product: multiply the two vectors element by element and add the results up. A large score means that the key fits the query well. `scale = 1 / sqrt(head_dim)` keeps the scores in a sensible range.

A small example with `head_dim = 2`, three keys, one-number values, and the scale left out:

```
q = [1, 0]

key j    k_j       s_j = q . k_j    exp(s_j)    p_j      v_j
  0      [ 1, 0]         1           2.718      0.665     10
  1      [ 0, 1]         0           1.000      0.245     20
  2      [-1, 0]        -1           0.368      0.090     30
                                     sum 4.086

out = 0.665 x 10 + 0.245 x 20 + 0.090 x 30 = 14.25
```

Key 0 fits best, so its value counts most. In the model each `v_j` is a vector of `head_dim` numbers, and so is `out`: every element of `out` is the same weighted average, taken over that element of the values.

As in task 2, the real softmax subtracts the largest score before calling `exp`, so that `exp` cannot overflow. The probabilities do not change.

## 2. Why the kernel works in tiles

The direct way takes three kernels: write all scores into a `[seq_q, seq_k]` matrix for each head, softmax each row, then multiply by the values. For one encoder layer that matrix is 20 x 1500 x 1500 floats, 180 MB, written once and read twice, although each score is needed only for a moment. Its size grows with the square of the sequence length.

Your kernel never stores the scores. Each query reads the keys 32 at a time (a tile), and keeps only a short running summary of what it has seen.

## 3. The running summary

A query keeps three running values:

```
m     the largest score seen so far
l     the sum of exp(s_j - m) over the keys seen so far
acc   the sum of exp(s_j - m) * v_j over the keys seen so far    (head_dim numbers)
```

`m` and `l` are the `MaxSum` of task 2; `acc` is new. When a tile brings a larger maximum `m_new`, everything already in `l` and `acc` was computed with the old `m` and is too large. Multiplying it by `corr = exp(m - m_new)`, a number between 0 and 1, fixes that. After the last tile, `out = acc / l`.

Here it is by hand, with four keys in two tiles of two and one-number values:

```
scores s:   1    2   |   3    0        values v:   10   20   |   30   40
            tile 1   |   tile 2

after tile 1:  m   = 2
               l   = exp(1-2) + exp(2-2)            = 0.368 + 1         = 1.368
               acc = 10 exp(1-2) + 20 exp(2-2)      = 3.68 + 20         = 23.68

tile 2:        m_new = 3, so corr = exp(2-3)        = 0.368
               exp(s - m_new) for the tile          = 1 and 0.050

after tile 2:  l   = 1.368 x 0.368 + (1 + 0.050)                        = 1.553
               acc = 23.68 x 0.368 + (30 x 1 + 40 x 0.050)              = 40.70
               m   = 3

out = acc / l = 40.70 / 1.553 = 26.21
```

The softmax of all four scores at once gives the same 26.21. If this is new to you, check it with a calculator.

## 4. Who does what

A block handles 4 neighbouring queries of one head, one query per warp:

```
grid: one block for every 4 queries, of every head, of every batch element

one block = 4 warps = 128 threads
+--------------+--------------+--------------+--------------+
| warp 0       | warp 1       | warp 2       | warp 3       |
| query 4b     | query 4b+1   | query 4b+2   | query 4b+3   |    b = blockIdx.x
+--------------+--------------+--------------+--------------+
| shared memory: the current tile of 32 keys and 32 values  |
+-----------------------------------------------------------+
```

The four warps share each tile, so every key and value is loaded from global memory once for four queries instead of once for each query.

Inside a warp, a lane has two jobs, one after the other:

```
scoring (4.3)         lane 0     lane 1     ...   lane 31
                      key j0     key j0+1         key j0+31          one dot product each

adding up (4.5)       lane 0     lane 1     ...   lane 31
                      dims 0,    dims 1,          dims 31,           head_dim = 128:
                      32, 64, 96 33, 65, 97       63, 95, 127        4 dims per lane
```

In between, the probabilities have to move between lanes: after scoring, the `p` of key `j0 + r` is in lane `r`, but for adding up, every lane needs all 32 of them.

One pass of the tile loop:

```
for each tile of 32 keys (j0 = 0, 32, 64, ...):
    +--------------------------------------------+
    | 4.2  copy the tile into shared memory,     |   all 128 threads
    |      then wait until everyone is done      |
    +--------------------------------------------+
    | 4.3  one score per lane                    |   each warp
    +--------------------------------------------+
    | 4.4  update m, l and acc                   |   each warp
    +--------------------------------------------+
    | 4.5  add the tile's values to acc          |   each warp
    +--------------------------------------------+
    |      wait again (given)                    |   all 128 threads
    +--------------------------------------------+
after the loop:
    4.6  out = acc / l
```

## 5. Where the data is

`q` and `out` are `[batch, seq_q, num_heads, head_dim]`; `k` and `v` are `[batch, seq_k, num_kv_heads, head_dim]`, as the linear layers produce them. The given code sets up two pointers for you: key `j` of the right head starts at `k_head + j * kv_stride`, and value `j` at `v_head + j * kv_stride`. Each warp's query is already copied into shared memory as `qv`, multiplied by `scale`.

The kernel's shared memory is one array, `smem`, cut into three parts:

```
+-------------------------------+---------------------------+---------------------+
| k_tile                        | v_tile                    | q_all               |
| 32 rows of head_dim + 1       | 32 rows of head_dim       | 4 rows of head_dim  |
+-------------------------------+---------------------------+---------------------+
```

Row `r` of `k_tile` starts at `k_tile + r * k_pitch`, with `k_pitch = head_dim + 1`. The extra float at the end of each row is padding against bank conflicts, for the same reason as the `+1` in task 3. Rows of `v_tile` have no padding.

## 6. Two special cases

### Grouped-query attention (GQA)

The text decoder has 16 query heads but only 4 key/value heads, and groups of neighbouring query heads share one:

```
query head:   0  1  2  3 |  4  5  6  7 |  8  9 10 11 | 12 13 14 15
K/V head:          0     |      1      |      2      |      3
```

In the audio encoder both counts are 20, so every query head has its own key/value head.

### Causal masking

In the decoder, a token may only look at itself and the tokens before it:

```
             key 0  1  2  3
query 0          x  .  .  .        x  may look
query 1          x  x  .  .        .  masked: score -INFINITY, so p = 0
query 2          x  x  x  .
query 3          x  x  x  x
```

The tests also have cases with fewer queries than keys, as in generation with a cache. The queries are then the last positions of the sequence. The given code works out `last_key`, the last key this warp's query may see.

The tile loop runs up to `block_last_key`, the last key that any query of the block needs, and not up to each warp's own `last_key`. Every thread of a block has to go round the loop the same number of times, because the loop contains `__syncthreads()`.

## 7. The blanks

| Blank | What it must do |
|-------|-----------------|
| 4.1 | Set `kv_h`, the K/V head that query head `h` reads. |
| 4.2 | Copy keys and values `j0 .. j0+31` of this head into `k_tile` and `v_tile`, with 0 for keys past `seq_k`. All threads of the block take part. |
| 4.3 | Set `s`, this lane's score for key `j0 + lane`, if the query may see that key. |
| 4.4 | Update `m`, `l` and `acc` with this tile, as in section 3, and set `p`. |
| 4.5 | Add the tile's values, weighted by `p`, to `acc`. |
| 4.6 | Write `acc / l` to `out`. |

## Test

```
make test4
```

When every case passes, the test ends with `Task 4 (attention): ALL PASS`.

The CPU checks every query of the small cases and about 300 queries of the large ones. The "peaky scores" case has large queries, so the maximum often changes from one tile to the next; it fails if the `corr` rescaling is wrong.

Then run `./run.sh`. With all four tasks on your kernels, the transcription should pass.

## Try this

1. Change `ATTN_WARPS` to 1, 2, 8 and 16 and time the audio encoder case. How does the number of times each key is loaded from global memory change, and how does the time follow?
2. Set `k_pitch = head_dim` (no `+1`) and time the encoder case again. With `head_dim = 64`, work out which banks the 32 lanes hit when each lane reads element `d` of its own row of `k_tile`.
3. Compare your attention with PyTorch's: `./run.sh --cuda task1,task2,task3 --profile` runs attention with PyTorch and everything else on your kernels. PyTorch does build the full score matrix, but it computes `Q K^T` and `P V` with cuBLAS. Compare the attention line of both profiles on your GPU, and think about which parts of your kernel do the most work per number loaded from shared memory.

## Questions for the report

1. How much memory would the full score matrix of one encoder layer take (20 heads, 1500 x 1500, float32)? How much shared memory does one block of your kernel use instead?
2. Why must all threads of a block run the same number of tile iterations? What could go wrong if each warp stopped at its own `last_key`?
3. What is `corr` when a tile does not raise the maximum? When can `m_new` be `-INFINITY`?
4. With `ATTN_WARPS = 4`, how many times is each key of one encoder head loaded from global memory per layer? What limits how large `ATTN_WARPS` can usefully be?
5. `acc` is declared with `MAX_DIMS_PER_LANE` entries, although only `dims` of them are used. Why? Where can the compiler keep `acc` when the loops over it have a fixed length, and where when they do not?
